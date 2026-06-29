#include <onnx.h>

struct operator_pdata_t {
	int q_num_heads;
	int kv_num_heads;
	int update_rule;
	float scale;
};

static float la_load(struct onnx_tensor_t * t, size_t idx)
{
	switch(t->type)
	{
	case ONNX_TENSOR_TYPE_BFLOAT16:
		return bfloat16_to_float32(((uint16_t *)t->datas)[idx]);
	case ONNX_TENSOR_TYPE_FLOAT16:
		return float16_to_float32(((uint16_t *)t->datas)[idx]);
	case ONNX_TENSOR_TYPE_FLOAT32:
		return ((float *)t->datas)[idx];
	default:
		return 0.0f;
	}
}

static void la_store(struct onnx_tensor_t * t, size_t idx, float v)
{
	switch(t->type)
	{
	case ONNX_TENSOR_TYPE_BFLOAT16:
		((uint16_t *)t->datas)[idx] = float32_to_bfloat16(v);
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		((uint16_t *)t->datas)[idx] = float32_to_float16(v);
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		((float *)t->datas)[idx] = v;
		break;
	default:
		break;
	}
}

static int LinearAttention_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	char * update_rule;

	if((n->ninput >= 3) && (n->ninput <= 6) && (n->noutput == 2) && n->inputs[0] && n->inputs[1] && n->inputs[2])
	{
		pdat = onnx_malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->q_num_heads = onnx_attribute_read_int(n, "q_num_heads", 0);
			pdat->kv_num_heads = onnx_attribute_read_int(n, "kv_num_heads", 0);
			pdat->scale = onnx_attribute_read_float(n, "scale", 0.0f);
			update_rule = onnx_attribute_read_string(n, "update_rule", "gated_delta");
			if(onnx_strcmp(update_rule, "linear") == 0)
				pdat->update_rule = 0;
			else if(onnx_strcmp(update_rule, "gated") == 0)
				pdat->update_rule = 1;
			else if(onnx_strcmp(update_rule, "delta") == 0)
				pdat->update_rule = 2;
			else
				pdat->update_rule = 3;
			if((pdat->q_num_heads > 0) && (pdat->kv_num_heads > 0) && ((pdat->q_num_heads % pdat->kv_num_heads) == 0))
			{
				n->priv = pdat;
				return 1;
			}
			onnx_free(pdat);
		}
	}
	return 0;
}

static int LinearAttention_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		onnx_free(pdat);
	return 1;
}

static int LinearAttention_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * q = n->inputs[0];
	struct onnx_tensor_t * k = n->inputs[1];
	struct onnx_tensor_t * v = n->inputs[2];
	int dk, dv;
	int odims[3];
	int sdims[4];

	if((q->ndim != 3) || (k->ndim != 3) || (v->ndim != 3))
		return 0;
	dk = k->dims[2] / pdat->kv_num_heads;
	dv = v->dims[2] / pdat->kv_num_heads;
	odims[0] = q->dims[0];
	odims[1] = q->dims[1];
	odims[2] = pdat->q_num_heads * dv;
	sdims[0] = q->dims[0];
	sdims[1] = pdat->kv_num_heads;
	sdims[2] = dk;
	sdims[3] = dv;
	if(!onnx_tensor_reshape(n->outputs[0], odims, 3, q->type))
		return 0;
	return onnx_tensor_reshape(n->outputs[1], sdims, 4, q->type);
}

static float LinearAttention_decay(struct onnx_tensor_t * decay, int b, int t, int h, int d, int kvh, int dk)
{
	int last = decay->dims[2];

	if(last == kvh)
		return la_load(decay, ((size_t)b * decay->dims[1] + t) * last + h);
	return la_load(decay, ((size_t)b * decay->dims[1] + t) * last + h * dk + d);
}

static float LinearAttention_beta(struct onnx_tensor_t * beta, int b, int t, int h, int kvh)
{
	int last = beta->dims[2];

	if(last == 1)
		return la_load(beta, ((size_t)b * beta->dims[1] + t) * last);
	return la_load(beta, ((size_t)b * beta->dims[1] + t) * last + h);
}

static void LinearAttention_operator(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * q = n->inputs[0];
	struct onnx_tensor_t * k = n->inputs[1];
	struct onnx_tensor_t * v = n->inputs[2];
	struct onnx_tensor_t * past = ((n->ninput >= 4) ? n->inputs[3] : NULL);
	struct onnx_tensor_t * decay = ((n->ninput >= 5) ? n->inputs[4] : NULL);
	struct onnx_tensor_t * beta = ((n->ninput >= 6) ? n->inputs[5] : NULL);
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * present = n->outputs[1];
	int bsz = q->dims[0];
	int tsz = q->dims[1];
	int qh = pdat->q_num_heads;
	int kvh = pdat->kv_num_heads;
	int dk = k->dims[2] / kvh;
	int dv = v->dims[2] / kvh;
	int group = qh / kvh;
	float scale = (pdat->scale == 0.0f) ? (1.0f / sqrtf((float)dk)) : pdat->scale;
	size_t state_len = (size_t)kvh * dk * dv;
	float * state = onnx_malloc(sizeof(float) * state_len);

	if(!state)
		return;
	for(int b = 0; b < bsz; b++)
	{
		for(size_t i = 0; i < state_len; i++)
			state[i] = past ? la_load(past, (size_t)b * state_len + i) : 0.0f;
		for(int t = 0; t < tsz; t++)
		{
			for(int h = 0; h < kvh; h++)
			{
				float gate[dk];
				float prediction[dv];
				float betav = beta ? LinearAttention_beta(beta, b, t, h, kvh) : 1.0f;
				for(int di = 0; di < dk; di++)
					gate[di] = ((pdat->update_rule == 1) || (pdat->update_rule == 3)) && decay ? expf(LinearAttention_decay(decay, b, t, h, di, kvh, dk)) : 1.0f;
				if((pdat->update_rule == 2) || (pdat->update_rule == 3))
				{
					for(int dj = 0; dj < dv; dj++)
					{
						float p = 0.0f;
						for(int di = 0; di < dk; di++)
						{
							float kval = la_load(k, ((size_t)b * tsz + t) * kvh * dk + h * dk + di);
							p += gate[di] * state[((size_t)h * dk + di) * dv + dj] * kval;
						}
						prediction[dj] = p;
					}
				}
				for(int di = 0; di < dk; di++)
				{
					float kval = la_load(k, ((size_t)b * tsz + t) * kvh * dk + h * dk + di);
					for(int dj = 0; dj < dv; dj++)
					{
						float vval = la_load(v, ((size_t)b * tsz + t) * kvh * dv + h * dv + dj);
						size_t sidx = ((size_t)h * dk + di) * dv + dj;
						if((pdat->update_rule == 0) || (pdat->update_rule == 1))
							state[sidx] = gate[di] * state[sidx] + kval * vval;
						else
							state[sidx] = gate[di] * state[sidx] + betav * kval * (vval - prediction[dj]);
					}
				}
			}
			for(int hq = 0; hq < qh; hq++)
			{
				int h = hq / group;
				for(int dj = 0; dj < dv; dj++)
				{
					float sum = 0.0f;
					for(int di = 0; di < dk; di++)
					{
						float qval = la_load(q, ((size_t)b * tsz + t) * qh * dk + hq * dk + di);
						sum += qval * state[((size_t)h * dk + di) * dv + dj];
					}
					la_store(y, ((size_t)b * tsz + t) * qh * dv + hq * dv + dj, sum * scale);
				}
			}
		}
		for(size_t i = 0; i < state_len; i++)
			la_store(present, (size_t)b * state_len + i, state[i]);
	}
	onnx_free(state);
}

void resolver_default_op_LinearAttention(struct onnx_node_t * n)
{
	if(n->opset >= 27)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BFLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = LinearAttention_init;
			n->exit = LinearAttention_exit;
			n->reshape = LinearAttention_reshape;
			n->op = LinearAttention_operator;
			break;
		default:
			break;
		}
	}
}
