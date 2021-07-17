import cv2
from openvino.inference_engine import IECore, IENetwork
import numpy as np

cpu_ext_dll = "C:/Program Files (x86)/IntelSWTools/openvino_2019.3.379/deployment_tools/inference_engine/bin/intel64/Release/cpu_extension_avx2.dll"


def load_to_IE(model):
	#1. feeding a model
	#getting bin file from the location that is remove xml and plce bin in extension(-3 from string and +bin)
	model_bin = model[:-3]+"bin"

	#loading the inference engine API
	ie = IECore()

	#loading IR file
	net = IENetwork(model = model, weights = model_bin)

	#2. checking for unsupported layers
	#listing all layers and supported layers
	cpu_extension_needed = False
	network_layers = net.layers.keys()
	supported_layer_map = ie.query_network(network = net, device_name = "CPU")
	supported_layers = supported_layer_map.keys()

	for layer in network_layers:
		if layer in supported_layers:
			pass
		else:
			cpu_extension_needed =True
			print("CPU extension needed")
			break

	# Adding CPU extension if needed
	if cpu_extension_needed:
		ie.add_extension(extension_path=cpu_ext, device_name="CPU")
		print("CPU extension added")
	else:
		print("CPU extension not needed")
	
	# Checking for any unsupported layers, if yes, exit
	supported_layer_map = ie.query_network(network=net, device_name="CPU")
	supported_layers = supported_layer_map.keys()
	unsupported_layer_exists = False
	network_layers = net.layers.keys()
	for layer in network_layers:
		if layer in supported_layers:
			pass
		else:
			print(layer +' : Still Unsupported')
			unsupported_layer_exists = True
	if unsupported_layer_exists:
		print("Exiting the program.")
		exit(1)
	
	# Loading the network to the inference engine
	exec_net = ie.load_network(network=net, device_name="CPU")
	print("IR successfully loaded into Inference Engine.")

	return exec_net


#synchronous inference - waits for the inference to give result before processing anouther input 
def sync_inference(exec_net, image):
	input_blob = next(iter(exec_net.inputs))
	result = exec_net.infer({input_blob:image})
	return result

#asynchronous inference - doesn't wait for inference to give result, rather processes other inputs
#request is given accoring to the no. of inputs
def async_inference(exec_net, image, request_id = 0): #request id is 0 because we only process 1 photo in this example
	input_blob = next(iter(exec_net.inputs))
	exec_net.start_async(request_id, inputs={input_blob: image})
	return exec_net


#sync_inference gives direct result
#to obtain async_inference's result
def get_async_output(exec_net, request_id = 0):
	output_blob = next(iter(exec_net.inputs))		#The “wait” function returns the status of the processing.
	status = exec_net.requests[request_id].wait(-1) #If we call the function with the argument 0, it will instantly 
													#return the status, even if the processing is not complete.
													#But if we call it with -1 as argument, it will wait for the process to complete.
	if status == 0:
		result = exec_net.requests[request_id].outputs[output_blob]
		return result


def preprocessing(input_image, height, width):
	#resize the image
	try:
		image = cv2.resize(input_image, (width,height))
	except:
		#if no face detected image will be null and to correct it make empty array of width*height
		image = np.zeros(width,height)
	#model expects color channel first and then dimension of image for eg(1*3*300*300) but opencv puts channel at last as (300*300*3)
	#transpose so that color channel comes first
	image = image.transpose((2,0,1))

	#adding the batch size model's list first argument i.e. 1
	image = image.reshape(1,3, height, width)

	return image


def get_input_shape(model):
	"""GIven a model, returns its input shape"""
	model_bin = model[:-3]+"bin"
	net = IENetwork(model=model, weights = model_bin)
	input_blob = next(iter(net.inputs))
	return net.inputs[input_blob].shape


