"""List input dims, RNN layers and dense-layer shapes of the legacy ONNX policies (read-only).
Usage: python onnx_inventory.py <dir-with-onnx-files>"""
import glob, os, sys
import onnx
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.onnx"))):
    m = onnx.load(f)
    ins = {i.name: [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim] for i in m.graph.input}
    inits = {i.name: list(i.dims) for i in m.graph.initializer}
    rnn = [(n.op_type, inits.get(n.input[1])) for n in m.graph.node if n.op_type in ("LSTM", "GRU")]
    dense = [s for s in inits.values() if len(s) == 2]
    print(f"{os.path.basename(f):34s} obs={ins['obs'][1]:>4}  rnn={rnn}  dense={dense}")
