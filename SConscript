from building import *
import rtconfig

# get current directory
cwd     = GetCurrentDir()
# The set of source files associated with this SConscript file.
src     = Glob('src/*.c')

if GetDepend('ONNX_BACKEND_USING_MNIST_EXAMPLE'):
	src    += Glob('examples/mnist.c')

if GetDepend('ONNX_BACKEND_USING_MNIST_SMALL_EXAMPLE'):
	src    += Glob('examples/mnist_sm.c')

if GetDepend('ONNX_BACKEND_USING_MNIST_MODEL_EXAMPLE'):
	src    += Glob('examples/mnist_model.c')

path   =  [cwd + '/src']
path   += [cwd + '/examples']

LOCAL_CCFLAGS = ''

group = DefineGroup('onnx-backend', src, depend = ['PKG_USING_ONNX_BACKEND'], CPPPATH = path, LOCAL_CCFLAGS = LOCAL_CCFLAGS)

Return('group')
