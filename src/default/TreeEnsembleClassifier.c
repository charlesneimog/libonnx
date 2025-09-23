#include <onnx.h>

#ifndef LIKELY
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

typedef struct {
    int M;
    int *feature;
    double *thr;
    uint8_t *mode;      /* 0=LEAF, 1=LEQ,2=LT,3=GTE,4=GT,5=EQ,6=NEQ */
    uint8_t *miss_true; /* NaN => true if 1 */
    int *true_idx;
    int *false_idx;
    int *roots; /* typically size 1 */
    int R;
    int *leaf_ofs; /* length M+1 */
    int *cls_ids;  /* pooled class ids for leaves */
    double *cls_w; /* pooled class weights for leaves */
    int pool_len;
} TreeEnsembleClassifier_Tree;

typedef enum {
    TreeEnsembleClassifier_POST_NONE = 0,
    TreeEnsembleClassifier_POST_SOFTMAX,
    TreeEnsembleClassifier_POST_LOGISTIC,
    TreeEnsembleClassifier_POST_PROBIT,
    TreeEnsembleClassifier_POST_SOFTMAX_ZERO
} tec_post_t;

typedef struct {
    int is_string_labels;
    int E;
    int64_t *labels_i64;
    ProtobufCBinaryData *labels_str;

    int F;
    int input_is_float;

    double *base_values;

    TreeEnsembleClassifier_Tree *trees;
    int K;

    double *row;
    double *scores;

    int last_cap;
    int *last_top;

    tec_post_t post;
} TreeEnsembleClassifier_Priv;

static Onnx__AttributeProto *TreeEnsembleClassifier_attr(Onnx__NodeProto *proto, const char *name) {
    if (!proto || !name) {
        return NULL;
    }
    for (size_t i = 0; i < proto->n_attribute; i++) {
        Onnx__AttributeProto *a = proto->attribute[i];
        if (a && a->name && onnx_strcmp(a->name, name) == 0) {
            return a;
        }
    }
    return NULL;
}

static inline int TreeEnsembleClassifier_mode_code_from_bin(const ProtobufCBinaryData *bin) {
    if (!bin) {
        return 1; /* default LEQ */
    }
    if (bin->len == 4 && onnx_memcmp(bin->data, "LEAF", 4) == 0) {
        return 0;
    }
    if (bin->len == 10 && onnx_memcmp(bin->data, "BRANCH_LEQ", 10) == 0) {
        return 1;
    }
    if (bin->len == 9 && onnx_memcmp(bin->data, "BRANCH_LT", 9) == 0) {
        return 2;
    }
    if (bin->len == 10 && onnx_memcmp(bin->data, "BRANCH_GTE", 10) == 0) {
        return 3;
    }
    if (bin->len == 9 && onnx_memcmp(bin->data, "BRANCH_GT", 9) == 0) {
        return 4;
    }
    if (bin->len == 9 && onnx_memcmp(bin->data, "BRANCH_EQ", 9) == 0) {
        return 5;
    }
    if (bin->len == 10 && onnx_memcmp(bin->data, "BRANCH_NEQ", 10) == 0) {
        return 6;
    }
    return 1;
}

static inline int TreeEnsembleClassifier_take_true_branch(uint8_t mode, double v, double thr) {
    switch (mode) {
    case 0:
        return 0; /* LEAF */
    case 1:
        return v <= thr; /* LEQ */
    case 2:
        return v < thr; /* LT  */
    case 3:
        return v >= thr; /* GTE */
    case 4:
        return v > thr; /* GT  */
    case 5:
        return v == thr; /* EQ  */
    case 6:
        return v != thr; /* NEQ */
    default:
        return v <= thr;
    }
}

static inline void TreeEnsembleClassifier_argmax(const double *v, int n, int *idx) {
    int mi = 0;
    double mb = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > mb) {
            mb = v[i];
            mi = i;
        }
    }
    *idx = mi;
}

static int cmp_pair_nodeid_asc(const void *a, const void *b) {
    const int64_t *pa = (const int64_t *)a;
    const int64_t *pb = (const int64_t *)b;
    if (pa[0] < pb[0]) {
        return -1;
    }
    if (pa[0] > pb[0]) {
        return 1;
    }
    return 0;
}

static int binsearch_i64(const int64_t *arr, int n, int64_t key) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        int64_t v = arr[mid];
        if (v < key) {
            lo = mid + 1;
        } else if (v > key) {
            hi = mid - 1;
        } else {
            return mid;
        }
    }
    return -1;
}

static inline int x_isnan_supported_type(const struct onnx_tensor_t *x) {
    return (x->type == ONNX_TENSOR_TYPE_FLOAT32 || x->type == ONNX_TENSOR_TYPE_FLOAT64);
}

static void TreeEnsembleClassifier_load_row_as_double(const struct onnx_tensor_t *x, int N, int F,
                                                      int r, double *buf) {
    (void)N;
    switch (x->type) {
    case ONNX_TENSOR_TYPE_FLOAT32: {
        const float *row = ((const float *)x->datas) + (size_t)r * (size_t)F;
        for (int f = 0; f < F; f++) {
            buf[f] = (double)row[f];
        }
        break;
    }
    case ONNX_TENSOR_TYPE_FLOAT64: {
        const double *row = ((const double *)x->datas) + (size_t)r * (size_t)F;
        for (int f = 0; f < F; f++) {
            buf[f] = row[f];
        }
        break;
    }
    case ONNX_TENSOR_TYPE_INT32: {
        const int32_t *row = ((const int32_t *)x->datas) + (size_t)r * (size_t)F;
        for (int f = 0; f < F; f++) {
            buf[f] = (double)row[f];
        }
        break;
    }
    case ONNX_TENSOR_TYPE_INT64: {
        const int64_t *row = ((const int64_t *)x->datas) + (size_t)r * (size_t)F;
        for (int f = 0; f < F; f++) {
            buf[f] = (double)row[f];
        }
        break;
    }
    default:
        for (int f = 0; f < F; f++) {
            buf[f] = 0.0;
        }
        break;
    }
}

static int TreeEnsembleClassifier_read_labels(struct onnx_node_t *n, int *is_str, int *E,
                                              int64_t **lab_i64, ProtobufCBinaryData **lab_str) {
    Onnx__AttributeProto *ai = TreeEnsembleClassifier_attr(n->proto, "classlabels_int64s");
    Onnx__AttributeProto *as = TreeEnsembleClassifier_attr(n->proto, "classlabels_strings");
    if (ai && ai->n_ints > 0) {
        *is_str = 0;
        *E = (int)ai->n_ints;
        *lab_i64 = ai->ints;
        *lab_str = NULL;
        return 1;
    }
    if (as && as->n_strings > 0) {
        *is_str = 1;
        *E = (int)as->n_strings;
        *lab_i64 = NULL;
        *lab_str = as->strings;
        return 1;
    }
    return 0;
}

static void TreeEnsembleClassifier_warn(struct onnx_node_t *n, const char *msg) {
    onnx_printf("\033[45;37mTreeEnsembleClassifier\033[0m: %s => %s-%d (%s)\r\n", msg,
                n->proto->op_type, n->opset,
                (onnx_strlen(n->proto->domain) > 0) ? n->proto->domain : "ai.onnx.ml");
}

static tec_post_t TreeEnsembleClassifier_decode_post(const char *post) {
    if (!post || onnx_strcmp(post, "NONE") == 0) {
        return TreeEnsembleClassifier_POST_NONE;
    }
    if (onnx_strcmp(post, "SOFTMAX") == 0) {
        return TreeEnsembleClassifier_POST_SOFTMAX;
    }
    if (onnx_strcmp(post, "LOGISTIC") == 0) {
        return TreeEnsembleClassifier_POST_LOGISTIC;
    }
    if (onnx_strcmp(post, "PROBIT") == 0) {
        return TreeEnsembleClassifier_POST_PROBIT;
    }
    if (onnx_strcmp(post, "SOFTMAX_ZERO") == 0) {
        return TreeEnsembleClassifier_POST_SOFTMAX_ZERO;
    }
    return TreeEnsembleClassifier_POST_NONE;
}

static inline void TreeEnsembleClassifier_apply_post_transform_enum(tec_post_t post, double *scores,
                                                                    int E) {
    switch (post) {
    case TreeEnsembleClassifier_POST_NONE:
        return;
    case TreeEnsembleClassifier_POST_SOFTMAX: {
        double m = scores[0];
        for (int i = 1; i < E; i++) {
            if (scores[i] > m) {
                m = scores[i];
            }
        }
        double s = 0.0;
        for (int i = 0; i < E; i++) {
            scores[i] = exp(scores[i] - m);
            s += scores[i];
        }
        if (s > 0.0) {
            for (int i = 0; i < E; i++) {
                scores[i] /= s;
            }
        }
        return;
    }
    case TreeEnsembleClassifier_POST_LOGISTIC:
        for (int i = 0; i < E; i++) {
            double x = scores[i];
            if (x >= 0) {
                double z = exp(-x);
                scores[i] = 1.0 / (1.0 + z);
            } else {
                double z = exp(x);
                scores[i] = z / (1.0 + z);
            }
        }
        return;
    case TreeEnsembleClassifier_POST_PROBIT: {
        const double invsqrt2 = 0.70710678118654752440;
        for (int i = 0; i < E; i++) {
            scores[i] = 0.5 * (1.0 + erf(scores[i] * invsqrt2));
        }
        return;
    }
    case TreeEnsembleClassifier_POST_SOFTMAX_ZERO: {
        double m = 0.0;
        for (int i = 0; i < E; i++) {
            if (scores[i] > m) {
                m = scores[i];
            }
        }
        double s = 1.0; /* implicit 0-score class */
        for (int i = 0; i < E; i++) {
            scores[i] = exp(scores[i] - m);
            s += scores[i];
        }
        if (s > 0.0) {
            for (int i = 0; i < E; i++) {
                scores[i] /= s;
            }
        }
        return;
    }
    }
}

static int TreeEnsembleClassifier_init(struct onnx_node_t *n) {
    if (!n || n->ninput < 1 || !n->inputs[0]) {
        return 0;
    }
    const struct onnx_tensor_t *x = n->inputs[0];
    if (!(x->ndim == 2 || x->ndim == 1)) {
        return 0;
    }
    if (!(x->type == ONNX_TENSOR_TYPE_FLOAT32 || x->type == ONNX_TENSOR_TYPE_FLOAT64 ||
          x->type == ONNX_TENSOR_TYPE_INT32 || x->type == ONNX_TENSOR_TYPE_INT64)) {
        TreeEnsembleClassifier_warn(n, "unsupported input type");
        return 0;
    }

    TreeEnsembleClassifier_Priv *P = (TreeEnsembleClassifier_Priv *)onnx_malloc(sizeof(*P));
    if (!P) {
        return 0;
    }
    onnx_memset(P, 0, sizeof(*P));
    n->priv = P;

    if (!TreeEnsembleClassifier_read_labels(n, &P->is_string_labels, &P->E, &P->labels_i64,
                                            &P->labels_str) ||
        P->E <= 0) {
        TreeEnsembleClassifier_warn(n, "missing classlabels_*");
        return 0;
    }

    int F = (x->ndim == 2) ? x->dims[1] : x->dims[0];
    if (F <= 0) {
        TreeEnsembleClassifier_warn(n, "invalid input feature dimension");
        return 0;
    }
    P->F = F;
    P->input_is_float = x_isnan_supported_type(x);

    P->post =
        TreeEnsembleClassifier_decode_post(onnx_attribute_read_string(n, "post_transform", "NONE"));

    P->base_values = (double *)onnx_malloc(sizeof(double) * (size_t)P->E);
    if (!P->base_values) {
        return 0;
    }
    for (int i = 0; i < P->E; i++) {
        P->base_values[i] = 0.0;
    }
    float *bv_f = NULL;
    int nbv = onnx_attribute_read_floats(n, "base_values", &bv_f);
    if (nbv > 0) {
        for (int i = 0; i < P->E; i++) {
            P->base_values[i] = (i < nbv) ? (double)bv_f[i] : 0.0;
        }
    }

    int64_t *nodes_treeids = NULL, *nodes_nodeids = NULL, *nodes_featureids = NULL;
    int64_t *nodes_truenodeids = NULL, *nodes_falsenodeids = NULL;
    float *nodes_values_f = NULL;
    Onnx__AttributeProto *amodes = TreeEnsembleClassifier_attr(n->proto, "nodes_modes");
    Onnx__AttributeProto *amiss =
        TreeEnsembleClassifier_attr(n->proto, "nodes_missing_value_tracks_true");

    int L_tree = onnx_attribute_read_ints(n, "nodes_treeids", &nodes_treeids);
    int L_node = onnx_attribute_read_ints(n, "nodes_nodeids", &nodes_nodeids);
    int L_feat = onnx_attribute_read_ints(n, "nodes_featureids", &nodes_featureids);
    int L_true = onnx_attribute_read_ints(n, "nodes_truenodeids", &nodes_truenodeids);
    int L_false = onnx_attribute_read_ints(n, "nodes_falsenodeids", &nodes_falsenodeids);
    int L_values = onnx_attribute_read_floats(n, "nodes_values", &nodes_values_f);
    if (L_tree <= 0 || L_node <= 0 || L_feat <= 0 || L_true <= 0 || L_false <= 0 || !amodes ||
        amodes->n_strings != (size_t)L_tree) {
        TreeEnsembleClassifier_warn(n, "missing or inconsistent nodes_*");
        return 0;
    }
    size_t L = (size_t)L_tree;
    if ((size_t)L_node != L || (size_t)L_feat != L || (size_t)L_true != L || (size_t)L_false != L) {
        TreeEnsembleClassifier_warn(n, "nodes_* length mismatch");
        return 0;
    }

    int K = 0;
    int64_t *uniq = (int64_t *)onnx_malloc(sizeof(int64_t) * L);
    if (!uniq) {
        return 0;
    }
    for (size_t i = 0; i < L; i++) {
        int found = 0;
        for (int j = 0; j < K; j++) {
            if (uniq[j] == nodes_treeids[i]) {
                found = 1;
                break;
            }
        }
        if (!found) {
            uniq[K++] = nodes_treeids[i];
        }
    }
    P->K = K;
    P->trees =
        (TreeEnsembleClassifier_Tree *)onnx_malloc(sizeof(TreeEnsembleClassifier_Tree) * (size_t)K);
    if (!P->trees) {
        onnx_free(uniq);
        return 0;
    }
    onnx_memset(P->trees, 0, sizeof(TreeEnsembleClassifier_Tree) * (size_t)K);

    if (L_values <= 0) {
        TreeEnsembleClassifier_warn(n, "nodes_values is required");
        onnx_free(uniq);
        return 0;
    }
    double *nodes_values = (double *)onnx_malloc(sizeof(double) * L);
    if (!nodes_values) {
        onnx_free(uniq);
        return 0;
    }
    for (size_t i = 0; i < L; i++) {
        nodes_values[i] = (double)nodes_values_f[i];
    }

    uint8_t *miss_flags = (uint8_t *)onnx_malloc(L);
    if (!miss_flags) {
        onnx_free(nodes_values);
        onnx_free(uniq);
        return 0;
    }
    if (amiss && amiss->n_ints == L) {
        for (size_t i = 0; i < L; i++) {
            miss_flags[i] = (uint8_t)(amiss->ints[i] ? 1 : 0);
        }
    } else {
        onnx_memset(miss_flags, 0, L);
    }

    int64_t *class_treeids = NULL, *class_nodeids = NULL, *class_ids = NULL;
    float *class_weights = NULL;
    int LCt = onnx_attribute_read_ints(n, "class_treeids", &class_treeids);
    int LCn = onnx_attribute_read_ints(n, "class_nodeids", &class_nodeids);
    int LCi = onnx_attribute_read_ints(n, "class_ids", &class_ids);
    int LCw = onnx_attribute_read_floats(n, "class_weights", &class_weights);
    if (LCt <= 0 || LCn <= 0 || LCi <= 0 || LCw <= 0 || LCt != LCn || LCt != LCi || LCt != LCw) {
        TreeEnsembleClassifier_warn(n, "missing class_* arrays");
        onnx_free(miss_flags);
        onnx_free(nodes_values);
        onnx_free(uniq);
        return 0;
    }
    size_t LC = (size_t)LCt;

    for (int ti = 0; ti < K; ti++) {
        int64_t treeid = uniq[ti];

        int M = 0;
        for (size_t i = 0; i < L; i++) {
            if (nodes_treeids[i] == treeid) {
                M++;
            }
        }
        if (M <= 0) {
            continue;
        }

        int64_t (*pairs)[2] = (int64_t (*)[2])onnx_malloc(sizeof(int64_t) * 2 * (size_t)M);
        if (!pairs) {
            onnx_free(miss_flags);
            onnx_free(nodes_values);
            onnx_free(uniq);
            return 0;
        }
        int m = 0;
        for (size_t i = 0; i < L; i++) {
            if (nodes_treeids[i] == treeid) {
                pairs[m][0] = nodes_nodeids[i];
                pairs[m][1] = (int64_t)i;
                m++;
            }
        }
        qsort(pairs, (size_t)M, sizeof(pairs[0]), cmp_pair_nodeid_asc);

        TreeEnsembleClassifier_Tree *T = &P->trees[ti];
        T->M = M;
        T->feature = (int *)onnx_malloc(sizeof(int) * (size_t)M);
        T->thr = (double *)onnx_malloc(sizeof(double) * (size_t)M);
        T->mode = (uint8_t *)onnx_malloc((size_t)M);
        T->miss_true = (uint8_t *)onnx_malloc((size_t)M);
        T->true_idx = (int *)onnx_malloc(sizeof(int) * (size_t)M);
        T->false_idx = (int *)onnx_malloc(sizeof(int) * (size_t)M);
        T->leaf_ofs = (int *)onnx_malloc(sizeof(int) * ((size_t)M + 1));
        if (!T->feature || !T->thr || !T->mode || !T->miss_true || !T->true_idx || !T->false_idx ||
            !T->leaf_ofs) {
            onnx_free(pairs);
            onnx_free(miss_flags);
            onnx_free(nodes_values);
            onnx_free(uniq);
            return 0;
        }

        int64_t *nodeids_sorted = (int64_t *)onnx_malloc(sizeof(int64_t) * (size_t)M);
        if (!nodeids_sorted) {
            onnx_free(pairs);
            onnx_free(miss_flags);
            onnx_free(nodes_values);
            onnx_free(uniq);
            return 0;
        }
        for (int i = 0; i < M; i++) {
            nodeids_sorted[i] = pairs[i][0];
        }

        Onnx__AttributeProto *amodes_local = amodes; /* local alias */
        for (int i = 0; i < M; i++) {
            int gi = (int)pairs[i][1];
            T->feature[i] = (int)nodes_featureids[gi];
            T->thr[i] = nodes_values[gi];
            T->mode[i] =
                (uint8_t)TreeEnsembleClassifier_mode_code_from_bin(&amodes_local->strings[gi]);
            T->miss_true[i] = miss_flags[gi];

            int64_t t_id = nodes_truenodeids[gi];
            int64_t f_id = nodes_falsenodeids[gi];
            int tii = (t_id >= 0) ? binsearch_i64(nodeids_sorted, M, t_id) : -1;
            int fii = (f_id >= 0) ? binsearch_i64(nodeids_sorted, M, f_id) : -1;
            T->true_idx[i] = (tii >= 0) ? tii : -1;
            T->false_idx[i] = (fii >= 0) ? fii : -1;
        }

        uint8_t *is_child = (uint8_t *)onnx_malloc((size_t)M);
        if (!is_child) {
            onnx_free(nodeids_sorted);
            onnx_free(pairs);
            onnx_free(miss_flags);
            onnx_free(nodes_values);
            onnx_free(uniq);
            return 0;
        }
        onnx_memset(is_child, 0, (size_t)M);
        for (int i = 0; i < M; i++) {
            if (T->true_idx[i] >= 0) {
                is_child[T->true_idx[i]] = 1;
            }
            if (T->false_idx[i] >= 0) {
                is_child[T->false_idx[i]] = 1;
            }
        }
        int R = 0;
        for (int i = 0; i < M; i++) {
            if (!is_child[i]) {
                R++;
            }
        }
        if (R <= 0) {
            R = 1;
        }
        T->roots = (int *)onnx_malloc(sizeof(int) * (size_t)R);
        if (!T->roots) {
            onnx_free(is_child);
            onnx_free(nodeids_sorted);
            onnx_free(pairs);
            onnx_free(miss_flags);
            onnx_free(nodes_values);
            onnx_free(uniq);
            return 0;
        }
        T->R = R;
        int q = 0;
        for (int i = 0; i < M && q < R; i++) {
            if (!is_child[i]) {
                T->roots[q++] = i;
            }
        }
        if (q == 0) {
            T->roots[0] = 0;
        }
        onnx_free(is_child);

        int *cnt = (int *)onnx_malloc(sizeof(int) * (size_t)M);
        if (!cnt) {
            onnx_free(nodeids_sorted);
            onnx_free(pairs);
            onnx_free(miss_flags);
            onnx_free(nodes_values);
            onnx_free(uniq);
            return 0;
        }
        for (int i = 0; i < M; i++) {
            cnt[i] = 0;
        }

        for (size_t j = 0; j < LC; j++) {
            if (class_treeids[j] != treeid) {
                continue;
            }
            int li = binsearch_i64(nodeids_sorted, M, class_nodeids[j]);
            if (li >= 0) {
                cnt[li]++;
            }
        }
        T->leaf_ofs[0] = 0;
        for (int i = 0; i < M; i++) {
            T->leaf_ofs[i + 1] = T->leaf_ofs[i] + cnt[i];
        }
        T->pool_len = T->leaf_ofs[M];
        T->cls_ids = (int *)onnx_malloc(sizeof(int) * (size_t)T->pool_len);
        T->cls_w = (double *)onnx_malloc(sizeof(double) * (size_t)T->pool_len);
        if (!T->cls_ids || !T->cls_w) {
            onnx_free(cnt);
            onnx_free(nodeids_sorted);
            onnx_free(pairs);
            onnx_free(miss_flags);
            onnx_free(nodes_values);
            onnx_free(uniq);
            return 0;
        }

        int *widx = (int *)onnx_malloc(sizeof(int) * (size_t)M);
        if (!widx) {
            onnx_free(cnt);
            onnx_free(nodeids_sorted);
            onnx_free(pairs);
            onnx_free(miss_flags);
            onnx_free(nodes_values);
            onnx_free(uniq);
            return 0;
        }
        for (int i = 0; i < M; i++) {
            widx[i] = T->leaf_ofs[i];
        }

        for (size_t j = 0; j < LC; j++) {
            if (class_treeids[j] != treeid) {
                continue;
            }
            int li = binsearch_i64(nodeids_sorted, M, class_nodeids[j]);
            if (li >= 0) {
                int pos = widx[li]++;
                T->cls_ids[pos] = (int)class_ids[j];
                T->cls_w[pos] = (double)class_weights[j];
            }
        }
        onnx_free(widx);
        onnx_free(cnt);
        onnx_free(nodeids_sorted);
        onnx_free(pairs);
    }

    onnx_free(miss_flags);
    onnx_free(nodes_values);
    onnx_free(uniq);

    P->row = (double *)onnx_malloc(sizeof(double) * (size_t)P->F);
    P->scores = (double *)onnx_malloc(sizeof(double) * (size_t)P->E);
    if (!P->row || !P->scores) {
        return 0;
    }

    P->last_cap = 0;
    P->last_top = NULL;

    return 1;
}

static int TreeEnsembleClassifier_exit(struct onnx_node_t *n) {
    if (!n || !n->priv) {
        return 1;
    }
    TreeEnsembleClassifier_Priv *P = (TreeEnsembleClassifier_Priv *)n->priv;
    if (P->trees) {
        for (int ti = 0; ti < P->K; ti++) {
            TreeEnsembleClassifier_Tree *T = &P->trees[ti];
            if (T->feature) {
                onnx_free(T->feature);
            }
            if (T->thr) {
                onnx_free(T->thr);
            }
            if (T->mode) {
                onnx_free(T->mode);
            }
            if (T->miss_true) {
                onnx_free(T->miss_true);
            }
            if (T->true_idx) {
                onnx_free(T->true_idx);
            }
            if (T->false_idx) {
                onnx_free(T->false_idx);
            }
            if (T->leaf_ofs) {
                onnx_free(T->leaf_ofs);
            }
            if (T->cls_ids) {
                onnx_free(T->cls_ids);
            }
            if (T->cls_w) {
                onnx_free(T->cls_w);
            }
            if (T->roots) {
                onnx_free(T->roots);
            }
        }
        onnx_free(P->trees);
    }
    if (P->base_values) {
        onnx_free(P->base_values);
    }
    if (P->row) {
        onnx_free(P->row);
    }
    if (P->scores) {
        onnx_free(P->scores);
    }
    if (P->last_top) {
        onnx_free(P->last_top);
    }
    onnx_free(P);
    n->priv = NULL;
    return 1;
}

static int TreeEnsembleClassifier_reshape(struct onnx_node_t *n) {
    TreeEnsembleClassifier_Priv *P = (TreeEnsembleClassifier_Priv *)n->priv;
    if (!P) {
        return 0;
    }
    struct onnx_tensor_t *x = n->inputs[0];
    int N = (x->ndim == 2) ? x->dims[0] : 1;

    if (n->noutput >= 1 && n->outputs[0]) {
        int d1[1] = {N};
        if (P->is_string_labels) {
            onnx_tensor_reshape(n->outputs[0], d1, 1, ONNX_TENSOR_TYPE_STRING);
        } else {
            onnx_tensor_reshape(n->outputs[0], d1, 1, ONNX_TENSOR_TYPE_INT64);
        }
    }
    /* Z: [N, E] */
    if (n->noutput >= 2 && n->outputs[1]) {
        int d2[2] = {N, P->E};
        onnx_tensor_reshape(n->outputs[1], d2, 2, ONNX_TENSOR_TYPE_FLOAT32);
    }

    if (N > P->last_cap) {
        int *nt = (int *)onnx_malloc(sizeof(int) * (size_t)N);
        if (!nt) {
            return 0;
        }
        for (int i = 0; i < N; i++) {
            nt[i] = -1;
        }
        if (P->last_top) {
            onnx_free(P->last_top);
        }
        P->last_top = nt;
        P->last_cap = N;
    } else {
        for (int i = 0; i < N; i++) {
            P->last_top[i] = -1;
        }
    }
    return 1;
}

static inline void TreeEnsembleClassifier_write_label_if_changed(struct onnx_tensor_t *y, int r,
                                                                 int is_str, int64_t *lab_i64,
                                                                 ProtobufCBinaryData *lab_str,
                                                                 int cls_idx, int *last_top) {
    if (last_top && last_top[r] == cls_idx) {
        return;
    }
    if (is_str) {
        char **out = (char **)y->datas;
        if (out[r]) {
            onnx_free(out[r]);
            out[r] = NULL;
        }
        size_t len = lab_str[cls_idx].len;
        char *s = (char *)onnx_malloc(len + 1);
        if (s) {
            onnx_memcpy(s, lab_str[cls_idx].data, len);
            s[len] = '\0';
        }
        out[r] = s;
    } else {
        int64_t *out = (int64_t *)y->datas;
        out[r] = lab_i64[cls_idx];
    }
    if (last_top) {
        last_top[r] = cls_idx;
    }
}

static void TreeEnsembleClassifier_operator(struct onnx_node_t *n) {
    TreeEnsembleClassifier_Priv *P = (TreeEnsembleClassifier_Priv *)n->priv;
    if (!P) {
        return;
    }

    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *Y = (n->noutput >= 1) ? n->outputs[0] : NULL;
    struct onnx_tensor_t *Z = (n->noutput >= 2) ? n->outputs[1] : NULL;

    int N, F;
    if (x->ndim == 2) {
        N = x->dims[0];
        F = x->dims[1];
    } else {
        N = 1;
        F = x->dims[0];
    }

    if (UNLIKELY(F != P->F)) {
        TreeEnsembleClassifier_warn(n, "input feature dimension changed; results may be invalid");
    }

    const int E = P->E;
    const int input_nan_check = P->input_is_float;
    const tec_post_t post = P->post;

    for (int r = 0; r < N; r++) {
        TreeEnsembleClassifier_load_row_as_double(x, N, F, r, P->row);

        onnx_memcpy(P->scores, P->base_values, sizeof(double) * (size_t)E);

        /* Traverse all trees (tight inner loop with local aliases) */
        for (int ti = 0; ti < P->K; ti++) {
            TreeEnsembleClassifier_Tree *T = &P->trees[ti];
            if (UNLIKELY(T->M <= 0 || T->R <= 0 || !T->roots)) {
                continue;
            }

            const int M = T->M;
            const int *feature = T->feature;
            const double *thr = T->thr;
            const uint8_t *mode = T->mode;
            const uint8_t *miss_true = T->miss_true;
            const int *true_idx = T->true_idx;
            const int *false_idx = T->false_idx;
            const int *leaf_ofs = T->leaf_ofs;
            const int *cls_ids = T->cls_ids;
            const double *cls_w = T->cls_w;

            int idx = T->roots[0];
            for (;;) {
                if (UNLIKELY(idx < 0 || idx >= M)) {
                    break;
                }

                uint8_t m = mode[idx];
                if (m == 0) { /* LEAF */
                    int s = leaf_ofs[idx], e = leaf_ofs[idx + 1];
                    for (int p = s; p < e; p++) {
                        int cid = cls_ids[p];
                        if (LIKELY((unsigned)cid < (unsigned)E)) {
                            P->scores[cid] += cls_w[p];
                        }
                    }
                    break;
                }
                int fid = feature[idx];
                double v = (fid >= 0 && fid < F) ? P->row[fid] : 0.0;

                int go_true;
                if (input_nan_check && isnan(v)) {
                    go_true = miss_true[idx] ? 1 : 0;
                } else {
                    go_true = TreeEnsembleClassifier_take_true_branch(m, v, thr[idx]);
                }

                int next = go_true ? true_idx[idx] : false_idx[idx];
                if (UNLIKELY(next < 0 || next == idx)) {
                    break;
                }
                idx = next;
            }
        }

        TreeEnsembleClassifier_apply_post_transform_enum(post, P->scores, E);

        if (Z) {
            float *zout = (float *)Z->datas + (size_t)r * (size_t)E;
            for (int c = 0; c < E; c++) {
                zout[c] = (float)P->scores[c];
            }
        }

        int top;
        TreeEnsembleClassifier_argmax(P->scores, E, &top);
        if (Y) {
            TreeEnsembleClassifier_write_label_if_changed(Y, r, P->is_string_labels, P->labels_i64,
                                                          P->labels_str, top, P->last_top);
        }
    }
}

void resolver_default_op_TreeEnsembleClassifier(struct onnx_node_t *n) {
    if (!n || n->ninput < 1) {
        return;
    }
    struct onnx_tensor_t *x = n->inputs[0];
    if (!x) {
        return;
    }
    switch (x->type) {
    case ONNX_TENSOR_TYPE_FLOAT32:
    case ONNX_TENSOR_TYPE_FLOAT64:
    case ONNX_TENSOR_TYPE_INT32:
    case ONNX_TENSOR_TYPE_INT64:
        n->init = TreeEnsembleClassifier_init;
        n->exit = TreeEnsembleClassifier_exit;
        n->reshape = TreeEnsembleClassifier_reshape;
        n->op = TreeEnsembleClassifier_operator;
        break;
    default:
        break;
    }
}
