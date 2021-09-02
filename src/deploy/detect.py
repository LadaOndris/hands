from openvino.inference_engine import IECore, IENetwork
import numpy as np

xml_file = "/media/sf_downloads/shared/blazeface_handseg.xml"
bin_file = "/media/sf_downloads/shared/blazeface_handseg.bin"
target = 'CPU'

ie = IECore()
net = ie.read_network(model=xml_file, weights=bin_file)
config = {'PERF_COUNT': 'YES'}
exec_net = ie.load_network(network=net, device_name=target, config=config)

image = np.ones([1, 256, 256, 1])
infer_request_handle = exec_net.start_async(request_id=0, inputs={'input': image})
infer_status = infer_request_handle.wait()


pred_deltas = infer_request_handle.output_blobs['deltas'].buffer
pred_confs = infer_request_handle.output_blobs['confs'].buffer

print(pred_confs)
