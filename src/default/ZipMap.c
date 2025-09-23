#include <onnx.h>

typedef struct {
    int is_string_labels;
    int E;
    int64_t *labels_i64;
    ProtobufCBinaryData *labels_str;
} ZipMap_Priv;

static Onnx__AttributeProto *ZipMap_attr(Onnx__NodeProto *proto, const char *name) {
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

static int ZipMap_read_labels(struct onnx_node_t *n, int *is_str, int *E, int64_t **lab_i64,
                              ProtobufCBinaryData **lab_str) {
    Onnx__AttributeProto *ai = ZipMap_attr(n->proto, "classlabels_int64s");
    Onnx__AttributeProto *as = ZipMap_attr(n->proto, "classlabels_strings");
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

static void ZipMap_warn(struct onnx_node_t *n, const char *msg) {
    onnx_printf("\033[45;37mZipMap\033[0m: %s => %s-%d (%s)\r\n", msg, n->proto->op_type, n->opset,
                (onnx_strlen(n->proto->domain) > 0) ? n->proto->domain : "ai.onnx.ml");
}

static int ZipMap_init(struct onnx_node_t *n) {
    if (!n || n->ninput < 1 || !n->inputs[0]) {
        return 0;
    }

    ZipMap_Priv *P = (ZipMap_Priv *)onnx_malloc(sizeof(ZipMap_Priv));
    if (!P) {
        return 0;
    }
    onnx_memset(P, 0, sizeof(*P));
    n->priv = P;

    if (!ZipMap_read_labels(n, &P->is_string_labels, &P->E, &P->labels_i64, &P->labels_str) ||
        P->E <= 0) {
        ZipMap_warn(n, "missing classlabels_*");
        return 0;
    }

    /* Input must be float tensor by spec; we'll accept float32 or float64 and cast to float32. */
    enum onnx_tensor_type_t t = n->inputs[0]->type;
    if (!(t == ONNX_TENSOR_TYPE_FLOAT32 || t == ONNX_TENSOR_TYPE_FLOAT64)) {
        ZipMap_warn(n, "input tensor must be float32/float64; will attempt to run but results are "
                       "unspecified");
    }
    return 1;
}

static int ZipMap_exit(struct onnx_node_t *n) {
    if (!n) {
        return 1;
    }
    if (n->priv) {
        onnx_free(n->priv);
        n->priv = NULL;
    }
    return 1;
}

static int ZipMap_reshape(struct onnx_node_t *n) {
    if (!n || !n->priv) {
        return 0;
    }
    ZipMap_Priv *P = (ZipMap_Priv *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *z = (n->noutput >= 1) ? n->outputs[0] : NULL;
    if (!x || !z) {
        return 0;
    }

    int N = 1, E_in = 0;
    if (x->ndim == 2) {
        N = x->dims[0];
        E_in = x->dims[1];
    } else if (x->ndim == 1) {
        N = 1;
        E_in = x->dims[0];
    } else {
        N = 1;
        E_in = (int)x->ndata;
    }

    if (P->E != E_in) {
        ZipMap_warn(n, "number of keys does not match input columns");
    }

    int dims[2] = {N, P->E};
    return onnx_tensor_reshape(z, dims, 2, ONNX_TENSOR_TYPE_FLOAT32);
}

static void ZipMap_operator(struct onnx_node_t *n) {
    if (!n || !n->priv) {
        return;
    }
    ZipMap_Priv *P = (ZipMap_Priv *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *z = (n->noutput >= 1) ? n->outputs[0] : NULL;
    if (!x || !z) {
        return;
    }

    int N = 1, E_in = 0;
    if (x->ndim == 2) {
        N = x->dims[0];
        E_in = x->dims[1];
    } else if (x->ndim == 1) {
        N = 1;
        E_in = x->dims[0];
    } else {
        N = 1;
        E_in = (int)x->ndata;
    }

    const int E = P->E;
    const int C = XMIN(E_in, E);

    float *out = (float *)z->datas;

    switch (x->type) {
    case ONNX_TENSOR_TYPE_FLOAT32: {
        const float *in = (const float *)x->datas;
        for (int nrow = 0; nrow < N; nrow++) {
            const float *rin = in + (size_t)nrow * (size_t)E_in;
            float *rout = out + (size_t)nrow * (size_t)E;
            if (C > 0) {
                onnx_memcpy(rout, rin, sizeof(float) * (size_t)C);
            }
            if (E > C) {
                onnx_memset(rout + C, 0, sizeof(float) * (size_t)(E - C));
            }
        }
        break;
    }
    case ONNX_TENSOR_TYPE_FLOAT64: {
        const double *in = (const double *)x->datas;
        for (int nrow = 0; nrow < N; nrow++) {
            const double *rin = in + (size_t)nrow * (size_t)E_in;
            float *rout = out + (size_t)nrow * (size_t)E;
            for (int j = 0; j < C; j++) {
                rout[j] = (float)rin[j];
            }
            if (E > C) {
                onnx_memset(rout + C, 0, sizeof(float) * (size_t)(E - C));
            }
        }
        break;
    }
    default:
        /* Fallback: cast other numeric types if they ever appear */
        for (int nrow = 0; nrow < N; nrow++) {
            float *rout = out + (size_t)nrow * (size_t)E;
            onnx_memset(rout, 0, sizeof(float) * (size_t)E);
        }
        ZipMap_warn(n, "unsupported input type for ZipMap; output zeroed");
        break;
    }
}

void resolver_default_op_ZipMap(struct onnx_node_t *n) {
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
        n->init = ZipMap_init;
        n->exit = ZipMap_exit;
        n->reshape = ZipMap_reshape;
        n->op = ZipMap_operator;
        break;
    default:
        break;
    }
}
