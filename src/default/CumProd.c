#include <onnx.h>

struct operator_pdata_t {
	int exclusive;
	int reverse;
};

static int CumProd_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 2) && (n->noutput == 1) && n->inputs[0] && n->inputs[1] && onnx_tensor_is_scalar(n->inputs[1]))
	{
		pdat = onnx_malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->exclusive = onnx_attribute_read_int(n, "exclusive", 0) ? 1 : 0;
			pdat->reverse = onnx_attribute_read_int(n, "reverse", 0) ? 1 : 0;
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int CumProd_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		onnx_free(pdat);
	return 1;
}

static int CumProd_reshape(struct onnx_node_t * n)
{
	return onnx_tensor_reshape_identity(n->outputs[0], n->inputs[0], n->inputs[0]->type);
}

static int CumProd_axis(struct onnx_node_t * n)
{
	struct onnx_tensor_t * axis = n->inputs[1];
	int a = 0;

	if(axis->type == ONNX_TENSOR_TYPE_INT32)
		a = *((int32_t *)axis->datas);
	else if(axis->type == ONNX_TENSOR_TYPE_INT64)
		a = *((int64_t *)axis->datas);
	if(a < 0)
		a += n->inputs[0]->ndim;
	return a;
}

#define CUMPROD_DECLARE(name, type, one_value, load_expr, store_expr) \
static void CumProd_##name(struct onnx_node_t * n) \
{ \
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv; \
	struct onnx_tensor_t * x = n->inputs[0]; \
	struct onnx_tensor_t * y = n->outputs[0]; \
	type * px = (type *)x->datas; \
	type * py = (type *)y->datas; \
	int axis = CumProd_axis(n); \
	if((axis < 0) || (axis >= x->ndim)) \
		return; \
	int inner = x->strides[axis]; \
	int count = x->dims[axis]; \
	int outer = x->ndata / (count * inner); \
	for(int o = 0; o < outer; o++) \
	{ \
		for(int i = 0; i < inner; i++) \
		{ \
			double acc = (one_value); \
			if(pdat->reverse) \
			{ \
				for(int c = count - 1; c >= 0; c--) \
				{ \
					size_t idx = (size_t)o * count * inner + (size_t)c * inner + i; \
					double v = (load_expr); \
					if(pdat->exclusive) \
					{ \
						double out = acc; \
						acc *= v; \
						(store_expr); \
					} \
					else \
					{ \
						acc *= v; \
						double out = acc; \
						(store_expr); \
					} \
				} \
			} \
			else \
			{ \
				for(int c = 0; c < count; c++) \
				{ \
					size_t idx = (size_t)o * count * inner + (size_t)c * inner + i; \
					double v = (load_expr); \
					if(pdat->exclusive) \
					{ \
						double out = acc; \
						acc *= v; \
						(store_expr); \
					} \
					else \
					{ \
						acc *= v; \
						double out = acc; \
						(store_expr); \
					} \
				} \
			} \
		} \
	} \
}

CUMPROD_DECLARE(int32, int32_t, 1.0, (double)px[idx], py[idx] = (int32_t)out)
CUMPROD_DECLARE(int64, int64_t, 1.0, (double)px[idx], py[idx] = (int64_t)out)
CUMPROD_DECLARE(uint32, uint32_t, 1.0, (double)px[idx], py[idx] = (uint32_t)out)
CUMPROD_DECLARE(uint64, uint64_t, 1.0, (double)px[idx], py[idx] = (uint64_t)out)
CUMPROD_DECLARE(bfloat16, uint16_t, 1.0, (double)bfloat16_to_float32(px[idx]), py[idx] = float32_to_bfloat16((float)out))
CUMPROD_DECLARE(float16, uint16_t, 1.0, (double)float16_to_float32(px[idx]), py[idx] = float32_to_float16((float)out))
CUMPROD_DECLARE(float32, float, 1.0, (double)px[idx], py[idx] = (float)out)
CUMPROD_DECLARE(float64, double, 1.0, px[idx], py[idx] = out)

void resolver_default_op_CumProd(struct onnx_node_t * n)
{
	if(n->opset >= 26)
	{
		if((n->inputs[1]->type != ONNX_TENSOR_TYPE_INT32) && (n->inputs[1]->type != ONNX_TENSOR_TYPE_INT64))
			return;
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT32:
			n->init = CumProd_init; n->exit = CumProd_exit; n->reshape = CumProd_reshape; n->op = CumProd_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = CumProd_init; n->exit = CumProd_exit; n->reshape = CumProd_reshape; n->op = CumProd_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = CumProd_init; n->exit = CumProd_exit; n->reshape = CumProd_reshape; n->op = CumProd_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = CumProd_init; n->exit = CumProd_exit; n->reshape = CumProd_reshape; n->op = CumProd_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = CumProd_init; n->exit = CumProd_exit; n->reshape = CumProd_reshape; n->op = CumProd_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = CumProd_init; n->exit = CumProd_exit; n->reshape = CumProd_reshape; n->op = CumProd_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = CumProd_init; n->exit = CumProd_exit; n->reshape = CumProd_reshape; n->op = CumProd_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = CumProd_init; n->exit = CumProd_exit; n->reshape = CumProd_reshape; n->op = CumProd_float64;
			break;
		default:
			break;
		}
	}
}
