#include <onnx.h>

struct operator_pdata_t {
	int activation;
};

static float ccws_load(struct onnx_tensor_t * t, size_t idx)
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

static void ccws_store(struct onnx_tensor_t * t, size_t idx, float v)
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

static struct onnx_tensor_t * CausalConvWithState_bias(struct onnx_node_t * n)
{
	if((n->ninput >= 3) && n->inputs[2] && (n->inputs[2]->ndim == 1))
		return n->inputs[2];
	return NULL;
}

static struct onnx_tensor_t * CausalConvWithState_past(struct onnx_node_t * n)
{
	if((n->ninput >= 4) && n->inputs[3])
		return n->inputs[3];
	if((n->ninput >= 3) && n->inputs[2] && (n->inputs[2]->ndim == 3))
		return n->inputs[2];
	return NULL;
}

static int CausalConvWithState_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;
	char * activation;

	if((n->ninput >= 2) && (n->ninput <= 4) && (n->noutput == 2) && n->inputs[0] && n->inputs[1])
	{
		pdat = onnx_malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			activation = onnx_attribute_read_string(n, "activation", "none");
			pdat->activation = ((onnx_strcmp(activation, "silu") == 0) || (onnx_strcmp(activation, "swish") == 0)) ? 1 : 0;
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int CausalConvWithState_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		onnx_free(pdat);
	return 1;
}

static int CausalConvWithState_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * input = n->inputs[0];
	struct onnx_tensor_t * weight = n->inputs[1];
	int dims[3];

	if((input->ndim != 3) || (weight->ndim != 3))
		return 0;
	if(!onnx_tensor_reshape_identity(n->outputs[0], input, input->type))
		return 0;
	dims[0] = input->dims[0];
	dims[1] = input->dims[1];
	dims[2] = weight->dims[2] - 1;
	return onnx_tensor_reshape(n->outputs[1], dims, 3, input->type);
}

static void CausalConvWithState_operator(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * input = n->inputs[0];
	struct onnx_tensor_t * weight = n->inputs[1];
	struct onnx_tensor_t * bias = CausalConvWithState_bias(n);
	struct onnx_tensor_t * past = CausalConvWithState_past(n);
	struct onnx_tensor_t * output = n->outputs[0];
	struct onnx_tensor_t * present = n->outputs[1];
	int bsz = input->dims[0];
	int ch = input->dims[1];
	int len = input->dims[2];
	int k = weight->dims[2];
	int state = k - 1;

	for(int b = 0; b < bsz; b++)
	{
		for(int c = 0; c < ch; c++)
		{
			for(int t = 0; t < len; t++)
			{
				float sum = bias ? ccws_load(bias, c) : 0.0f;
				for(int j = 0; j < k; j++)
				{
					int src = t - (k - 1 - j);
					float xv = 0.0f;
					if(src >= 0)
						xv = ccws_load(input, ((size_t)b * ch + c) * len + src);
					else if(past)
					{
						int pidx = state + src;
						if((pidx >= 0) && (pidx < state))
							xv = ccws_load(past, ((size_t)b * ch + c) * state + pidx);
					}
					sum += xv * ccws_load(weight, ((size_t)c * k) + j);
				}
				if(pdat->activation)
					sum = sum / (1.0f + expf(-sum));
				ccws_store(output, ((size_t)b * ch + c) * len + t, sum);
			}
			for(int s = 0; s < state; s++)
			{
				int src = len - state + s;
				float v = 0.0f;
				if(src >= 0)
					v = ccws_load(input, ((size_t)b * ch + c) * len + src);
				else if(past)
				{
					int pidx = state + src;
					if((pidx >= 0) && (pidx < state))
						v = ccws_load(past, ((size_t)b * ch + c) * state + pidx);
				}
				ccws_store(present, ((size_t)b * ch + c) * state + s, v);
			}
		}
	}
}

void resolver_default_op_CausalConvWithState(struct onnx_node_t * n)
{
	if(n->opset >= 27)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BFLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = CausalConvWithState_init;
			n->exit = CausalConvWithState_exit;
			n->reshape = CausalConvWithState_reshape;
			n->op = CausalConvWithState_operator;
			break;
		default:
			break;
		}
	}
}
