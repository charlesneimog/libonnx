#include <onnx.h>

struct operator_pdata_t {
	enum onnx_tensor_type_t to;
};

static int BitCast_init(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninput == 1) && (n->noutput == 1) && n->inputs[0])
	{
		pdat = onnx_malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->to = (enum onnx_tensor_type_t)onnx_attribute_read_int(n, "to", ONNX_TENSOR_TYPE_UNDEFINED);
			if((pdat->to != ONNX_TENSOR_TYPE_UNDEFINED) && (pdat->to != ONNX_TENSOR_TYPE_STRING) &&
				(n->inputs[0]->type != ONNX_TENSOR_TYPE_STRING) &&
				(onnx_tensor_type_sizeof(pdat->to) == onnx_tensor_type_sizeof(n->inputs[0]->type)))
			{
				n->priv = pdat;
				return 1;
			}
			onnx_free(pdat);
		}
	}
	return 0;
}

static int BitCast_exit(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		onnx_free(pdat);
	return 1;
}

static int BitCast_reshape(struct onnx_node_t * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, pdat->to);
}

static void BitCast_operator(struct onnx_node_t * n)
{
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int sz = onnx_tensor_type_sizeof(x->type);

	if(sz > 0)
		onnx_memcpy(y->datas, x->datas, x->ndata * sz);
}

void resolver_default_op_BitCast(struct onnx_node_t * n)
{
	if(n->opset >= 26)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
		case ONNX_TENSOR_TYPE_INT4:
		case ONNX_TENSOR_TYPE_INT8:
		case ONNX_TENSOR_TYPE_INT16:
		case ONNX_TENSOR_TYPE_INT32:
		case ONNX_TENSOR_TYPE_INT64:
		case ONNX_TENSOR_TYPE_UINT4:
		case ONNX_TENSOR_TYPE_UINT8:
		case ONNX_TENSOR_TYPE_UINT16:
		case ONNX_TENSOR_TYPE_UINT32:
		case ONNX_TENSOR_TYPE_UINT64:
		case ONNX_TENSOR_TYPE_FLOAT4E2M1:
		case ONNX_TENSOR_TYPE_FLOAT8E4M3FN:
		case ONNX_TENSOR_TYPE_FLOAT8E4M3FNUZ:
		case ONNX_TENSOR_TYPE_FLOAT8E5M2:
		case ONNX_TENSOR_TYPE_FLOAT8E5M2FNUZ:
		case ONNX_TENSOR_TYPE_FLOAT8E8M0:
		case ONNX_TENSOR_TYPE_BFLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
		case ONNX_TENSOR_TYPE_COMPLEX64:
		case ONNX_TENSOR_TYPE_COMPLEX128:
			n->init = BitCast_init;
			n->exit = BitCast_exit;
			n->reshape = BitCast_reshape;
			n->op = BitCast_operator;
			break;
		default:
			break;
		}
	}
}
