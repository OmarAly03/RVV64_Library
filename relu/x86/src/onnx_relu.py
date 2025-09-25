import onnx
from onnx import helper, TensorProto

input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None]) 
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None]) 

relu_node = helper.make_node(
    "Relu",     # operator name
    inputs=["input"],
    outputs=["output"]
)

graph = helper.make_graph(
    nodes=[relu_node],
    name="ReLUActivationGraph",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

model = helper.make_model(graph, producer_name="relu_example", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7 

onnx.save(model, "../relu.onnx")
print("Saved relu.onnx")