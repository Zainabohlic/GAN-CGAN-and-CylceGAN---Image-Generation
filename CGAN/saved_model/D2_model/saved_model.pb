��	
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��
r
dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_43/bias
k
!dense_43/bias/Read/ReadVariableOpReadVariableOpdense_43/bias*
_output_shapes
:*
dtype0
|
dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_43/kernel
u
#dense_43/kernel/Read/ReadVariableOpReadVariableOpdense_43/kernel* 
_output_shapes
:
��*
dtype0
�
instance_normalization_339/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!instance_normalization_339/beta
�
3instance_normalization_339/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_339/beta*
_output_shapes	
:�*
dtype0
�
 instance_normalization_339/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" instance_normalization_339/gamma
�
4instance_normalization_339/gamma/Read/ReadVariableOpReadVariableOp instance_normalization_339/gamma*
_output_shapes	
:�*
dtype0
w
conv2d_307/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameconv2d_307/bias
p
#conv2d_307/bias/Read/ReadVariableOpReadVariableOpconv2d_307/bias*
_output_shapes	
:�*
dtype0
�
conv2d_307/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*"
shared_nameconv2d_307/kernel
�
%conv2d_307/kernel/Read/ReadVariableOpReadVariableOpconv2d_307/kernel*(
_output_shapes
:��*
dtype0
�
instance_normalization_338/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!instance_normalization_338/beta
�
3instance_normalization_338/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_338/beta*
_output_shapes	
:�*
dtype0
�
 instance_normalization_338/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" instance_normalization_338/gamma
�
4instance_normalization_338/gamma/Read/ReadVariableOpReadVariableOp instance_normalization_338/gamma*
_output_shapes	
:�*
dtype0
w
conv2d_306/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameconv2d_306/bias
p
#conv2d_306/bias/Read/ReadVariableOpReadVariableOpconv2d_306/bias*
_output_shapes	
:�*
dtype0
�
conv2d_306/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*"
shared_nameconv2d_306/kernel
�
%conv2d_306/kernel/Read/ReadVariableOpReadVariableOpconv2d_306/kernel*'
_output_shapes
:@�*
dtype0
v
conv2d_305/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_305/bias
o
#conv2d_305/bias/Read/ReadVariableOpReadVariableOpconv2d_305/bias*
_output_shapes
:@*
dtype0
�
conv2d_305/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_305/kernel

%conv2d_305/kernel/Read/ReadVariableOpReadVariableOpconv2d_305/kernel*&
_output_shapes
:@*
dtype0
�
serving_default_input_88Placeholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_88conv2d_305/kernelconv2d_305/biasconv2d_306/kernelconv2d_306/bias instance_normalization_338/gammainstance_normalization_338/betaconv2d_307/kernelconv2d_307/bias instance_normalization_339/gammainstance_normalization_339/betadense_43/kerneldense_43/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_386391

NoOpNoOp
�>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�=
value�=B�= B�=
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer-9
layer_with_weights-5
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses* 
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias
 +_jit_compiled_convolution_op*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
	2gamma
3beta*
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses* 
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias
 B_jit_compiled_convolution_op*
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
	Igamma
Jbeta*
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses* 
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

]kernel
^bias*
Z
0
1
)2
*3
24
35
@6
A7
I8
J9
]10
^11*
Z
0
1
)2
*3
24
35
@6
A7
I8
J9
]10
^11*
* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

dtrace_0
etrace_1* 

ftrace_0
gtrace_1* 
* 

hserving_default* 

0
1*

0
1*
* 
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
a[
VARIABLE_VALUEconv2d_305/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_305/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 

utrace_0* 

vtrace_0* 

)0
*1*

)0
*1*
* 
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

|trace_0* 

}trace_0* 
a[
VARIABLE_VALUEconv2d_306/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_306/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

20
31*

20
31*
* 
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
oi
VARIABLE_VALUE instance_normalization_338/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEinstance_normalization_338/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

@0
A1*

@0
A1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_307/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_307/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

I0
J1*

I0
J1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
oi
VARIABLE_VALUE instance_normalization_339/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEinstance_normalization_339/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

]0
^1*

]0
^1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_43/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_43/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
0
1
2
3
4
5
6
7
	8

9
10*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_305/kernelconv2d_305/biasconv2d_306/kernelconv2d_306/bias instance_normalization_338/gammainstance_normalization_338/betaconv2d_307/kernelconv2d_307/bias instance_normalization_339/gammainstance_normalization_339/betadense_43/kerneldense_43/biasConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_386707
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_305/kernelconv2d_305/biasconv2d_306/kernelconv2d_306/bias instance_normalization_338/gammainstance_normalization_338/betaconv2d_307/kernelconv2d_307/bias instance_normalization_339/gammainstance_normalization_339/betadense_43/kerneldense_43/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_386752��
�*
�
V__inference_instance_normalization_339_layer_call_and_return_conditional_losses_386155

inputs.
reshape_readvariableop_resource:	�0
!reshape_1_readvariableop_resource:	�
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(v
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:�����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:����������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(s
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:�*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:�w
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:�T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*0
_output_shapes
:����������f
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*0
_output_shapes
:����������v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:����������l
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*0
_output_shapes
:����������{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:����������x
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:����������{
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*0
_output_shapes
:����������k
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*0
_output_shapes
:����������V
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_386391
input_88!
unknown:@
	unknown_0:@$
	unknown_1:@�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_88unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_386005o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name386387:&"
 
_user_specified_name386385:&
"
 
_user_specified_name386383:&	"
 
_user_specified_name386381:&"
 
_user_specified_name386379:&"
 
_user_specified_name386377:&"
 
_user_specified_name386375:&"
 
_user_specified_name386373:&"
 
_user_specified_name386371:&"
 
_user_specified_name386369:&"
 
_user_specified_name386367:&"
 
_user_specified_name386365:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
input_88
�
�
;__inference_instance_normalization_338_layer_call_fn_386448

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *_
fZRX
V__inference_instance_normalization_338_layer_call_and_return_conditional_losses_386086x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name386444:&"
 
_user_specified_name386442:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
˳
�
!__inference__wrapped_model_386005
input_88L
2model_87_conv2d_305_conv2d_readvariableop_resource:@A
3model_87_conv2d_305_biasadd_readvariableop_resource:@M
2model_87_conv2d_306_conv2d_readvariableop_resource:@�B
3model_87_conv2d_306_biasadd_readvariableop_resource:	�R
Cmodel_87_instance_normalization_338_reshape_readvariableop_resource:	�T
Emodel_87_instance_normalization_338_reshape_1_readvariableop_resource:	�N
2model_87_conv2d_307_conv2d_readvariableop_resource:��B
3model_87_conv2d_307_biasadd_readvariableop_resource:	�R
Cmodel_87_instance_normalization_339_reshape_readvariableop_resource:	�T
Emodel_87_instance_normalization_339_reshape_1_readvariableop_resource:	�D
0model_87_dense_43_matmul_readvariableop_resource:
��?
1model_87_dense_43_biasadd_readvariableop_resource:
identity��*model_87/conv2d_305/BiasAdd/ReadVariableOp�)model_87/conv2d_305/Conv2D/ReadVariableOp�*model_87/conv2d_306/BiasAdd/ReadVariableOp�)model_87/conv2d_306/Conv2D/ReadVariableOp�*model_87/conv2d_307/BiasAdd/ReadVariableOp�)model_87/conv2d_307/Conv2D/ReadVariableOp�(model_87/dense_43/BiasAdd/ReadVariableOp�'model_87/dense_43/MatMul/ReadVariableOp�:model_87/instance_normalization_338/Reshape/ReadVariableOp�<model_87/instance_normalization_338/Reshape_1/ReadVariableOp�:model_87/instance_normalization_339/Reshape/ReadVariableOp�<model_87/instance_normalization_339/Reshape_1/ReadVariableOp�
)model_87/conv2d_305/Conv2D/ReadVariableOpReadVariableOp2model_87_conv2d_305_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
model_87/conv2d_305/Conv2DConv2Dinput_881model_87/conv2d_305/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
*model_87/conv2d_305/BiasAdd/ReadVariableOpReadVariableOp3model_87_conv2d_305_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_87/conv2d_305/BiasAddBiasAdd#model_87/conv2d_305/Conv2D:output:02model_87/conv2d_305/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @�
"model_87/leaky_re_lu_381/LeakyRelu	LeakyRelu$model_87/conv2d_305/BiasAdd:output:0*/
_output_shapes
:���������  @*
alpha%���>�
)model_87/conv2d_306/Conv2D/ReadVariableOpReadVariableOp2model_87_conv2d_306_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
model_87/conv2d_306/Conv2DConv2D0model_87/leaky_re_lu_381/LeakyRelu:activations:01model_87/conv2d_306/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
*model_87/conv2d_306/BiasAdd/ReadVariableOpReadVariableOp3model_87_conv2d_306_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_87/conv2d_306/BiasAddBiasAdd#model_87/conv2d_306/Conv2D:output:02model_87/conv2d_306/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
)model_87/instance_normalization_338/ShapeShape$model_87/conv2d_306/BiasAdd:output:0*
T0*
_output_shapes
::���
7model_87/instance_normalization_338/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9model_87/instance_normalization_338/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9model_87/instance_normalization_338/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1model_87/instance_normalization_338/strided_sliceStridedSlice2model_87/instance_normalization_338/Shape:output:0@model_87/instance_normalization_338/strided_slice/stack:output:0Bmodel_87/instance_normalization_338/strided_slice/stack_1:output:0Bmodel_87/instance_normalization_338/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9model_87/instance_normalization_338/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;model_87/instance_normalization_338/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;model_87/instance_normalization_338/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3model_87/instance_normalization_338/strided_slice_1StridedSlice2model_87/instance_normalization_338/Shape:output:0Bmodel_87/instance_normalization_338/strided_slice_1/stack:output:0Dmodel_87/instance_normalization_338/strided_slice_1/stack_1:output:0Dmodel_87/instance_normalization_338/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9model_87/instance_normalization_338/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;model_87/instance_normalization_338/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;model_87/instance_normalization_338/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3model_87/instance_normalization_338/strided_slice_2StridedSlice2model_87/instance_normalization_338/Shape:output:0Bmodel_87/instance_normalization_338/strided_slice_2/stack:output:0Dmodel_87/instance_normalization_338/strided_slice_2/stack_1:output:0Dmodel_87/instance_normalization_338/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9model_87/instance_normalization_338/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;model_87/instance_normalization_338/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;model_87/instance_normalization_338/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3model_87/instance_normalization_338/strided_slice_3StridedSlice2model_87/instance_normalization_338/Shape:output:0Bmodel_87/instance_normalization_338/strided_slice_3/stack:output:0Dmodel_87/instance_normalization_338/strided_slice_3/stack_1:output:0Dmodel_87/instance_normalization_338/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bmodel_87/instance_normalization_338/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
0model_87/instance_normalization_338/moments/meanMean$model_87/conv2d_306/BiasAdd:output:0Kmodel_87/instance_normalization_338/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
8model_87/instance_normalization_338/moments/StopGradientStopGradient9model_87/instance_normalization_338/moments/mean:output:0*
T0*0
_output_shapes
:�����������
=model_87/instance_normalization_338/moments/SquaredDifferenceSquaredDifference$model_87/conv2d_306/BiasAdd:output:0Amodel_87/instance_normalization_338/moments/StopGradient:output:0*
T0*0
_output_shapes
:�����������
Fmodel_87/instance_normalization_338/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
4model_87/instance_normalization_338/moments/varianceMeanAmodel_87/instance_normalization_338/moments/SquaredDifference:z:0Omodel_87/instance_normalization_338/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
:model_87/instance_normalization_338/Reshape/ReadVariableOpReadVariableOpCmodel_87_instance_normalization_338_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1model_87/instance_normalization_338/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
+model_87/instance_normalization_338/ReshapeReshapeBmodel_87/instance_normalization_338/Reshape/ReadVariableOp:value:0:model_87/instance_normalization_338/Reshape/shape:output:0*
T0*'
_output_shapes
:��
<model_87/instance_normalization_338/Reshape_1/ReadVariableOpReadVariableOpEmodel_87_instance_normalization_338_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3model_87/instance_normalization_338/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
-model_87/instance_normalization_338/Reshape_1ReshapeDmodel_87/instance_normalization_338/Reshape_1/ReadVariableOp:value:0<model_87/instance_normalization_338/Reshape_1/shape:output:0*
T0*'
_output_shapes
:�x
3model_87/instance_normalization_338/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1model_87/instance_normalization_338/batchnorm/addAddV2=model_87/instance_normalization_338/moments/variance:output:0<model_87/instance_normalization_338/batchnorm/add/y:output:0*
T0*0
_output_shapes
:�����������
3model_87/instance_normalization_338/batchnorm/RsqrtRsqrt5model_87/instance_normalization_338/batchnorm/add:z:0*
T0*0
_output_shapes
:�����������
1model_87/instance_normalization_338/batchnorm/mulMul7model_87/instance_normalization_338/batchnorm/Rsqrt:y:04model_87/instance_normalization_338/Reshape:output:0*
T0*0
_output_shapes
:�����������
3model_87/instance_normalization_338/batchnorm/mul_1Mul$model_87/conv2d_306/BiasAdd:output:05model_87/instance_normalization_338/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
3model_87/instance_normalization_338/batchnorm/mul_2Mul9model_87/instance_normalization_338/moments/mean:output:05model_87/instance_normalization_338/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
1model_87/instance_normalization_338/batchnorm/subSub6model_87/instance_normalization_338/Reshape_1:output:07model_87/instance_normalization_338/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:�����������
3model_87/instance_normalization_338/batchnorm/add_1AddV27model_87/instance_normalization_338/batchnorm/mul_1:z:05model_87/instance_normalization_338/batchnorm/sub:z:0*
T0*0
_output_shapes
:�����������
"model_87/leaky_re_lu_382/LeakyRelu	LeakyRelu7model_87/instance_normalization_338/batchnorm/add_1:z:0*0
_output_shapes
:����������*
alpha%���>�
)model_87/conv2d_307/Conv2D/ReadVariableOpReadVariableOp2model_87_conv2d_307_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model_87/conv2d_307/Conv2DConv2D0model_87/leaky_re_lu_382/LeakyRelu:activations:01model_87/conv2d_307/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
*model_87/conv2d_307/BiasAdd/ReadVariableOpReadVariableOp3model_87_conv2d_307_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_87/conv2d_307/BiasAddBiasAdd#model_87/conv2d_307/Conv2D:output:02model_87/conv2d_307/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
)model_87/instance_normalization_339/ShapeShape$model_87/conv2d_307/BiasAdd:output:0*
T0*
_output_shapes
::���
7model_87/instance_normalization_339/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9model_87/instance_normalization_339/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9model_87/instance_normalization_339/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1model_87/instance_normalization_339/strided_sliceStridedSlice2model_87/instance_normalization_339/Shape:output:0@model_87/instance_normalization_339/strided_slice/stack:output:0Bmodel_87/instance_normalization_339/strided_slice/stack_1:output:0Bmodel_87/instance_normalization_339/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9model_87/instance_normalization_339/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;model_87/instance_normalization_339/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;model_87/instance_normalization_339/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3model_87/instance_normalization_339/strided_slice_1StridedSlice2model_87/instance_normalization_339/Shape:output:0Bmodel_87/instance_normalization_339/strided_slice_1/stack:output:0Dmodel_87/instance_normalization_339/strided_slice_1/stack_1:output:0Dmodel_87/instance_normalization_339/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9model_87/instance_normalization_339/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;model_87/instance_normalization_339/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;model_87/instance_normalization_339/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3model_87/instance_normalization_339/strided_slice_2StridedSlice2model_87/instance_normalization_339/Shape:output:0Bmodel_87/instance_normalization_339/strided_slice_2/stack:output:0Dmodel_87/instance_normalization_339/strided_slice_2/stack_1:output:0Dmodel_87/instance_normalization_339/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9model_87/instance_normalization_339/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;model_87/instance_normalization_339/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;model_87/instance_normalization_339/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3model_87/instance_normalization_339/strided_slice_3StridedSlice2model_87/instance_normalization_339/Shape:output:0Bmodel_87/instance_normalization_339/strided_slice_3/stack:output:0Dmodel_87/instance_normalization_339/strided_slice_3/stack_1:output:0Dmodel_87/instance_normalization_339/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bmodel_87/instance_normalization_339/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
0model_87/instance_normalization_339/moments/meanMean$model_87/conv2d_307/BiasAdd:output:0Kmodel_87/instance_normalization_339/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
8model_87/instance_normalization_339/moments/StopGradientStopGradient9model_87/instance_normalization_339/moments/mean:output:0*
T0*0
_output_shapes
:�����������
=model_87/instance_normalization_339/moments/SquaredDifferenceSquaredDifference$model_87/conv2d_307/BiasAdd:output:0Amodel_87/instance_normalization_339/moments/StopGradient:output:0*
T0*0
_output_shapes
:�����������
Fmodel_87/instance_normalization_339/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
4model_87/instance_normalization_339/moments/varianceMeanAmodel_87/instance_normalization_339/moments/SquaredDifference:z:0Omodel_87/instance_normalization_339/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
:model_87/instance_normalization_339/Reshape/ReadVariableOpReadVariableOpCmodel_87_instance_normalization_339_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1model_87/instance_normalization_339/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
+model_87/instance_normalization_339/ReshapeReshapeBmodel_87/instance_normalization_339/Reshape/ReadVariableOp:value:0:model_87/instance_normalization_339/Reshape/shape:output:0*
T0*'
_output_shapes
:��
<model_87/instance_normalization_339/Reshape_1/ReadVariableOpReadVariableOpEmodel_87_instance_normalization_339_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3model_87/instance_normalization_339/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
-model_87/instance_normalization_339/Reshape_1ReshapeDmodel_87/instance_normalization_339/Reshape_1/ReadVariableOp:value:0<model_87/instance_normalization_339/Reshape_1/shape:output:0*
T0*'
_output_shapes
:�x
3model_87/instance_normalization_339/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1model_87/instance_normalization_339/batchnorm/addAddV2=model_87/instance_normalization_339/moments/variance:output:0<model_87/instance_normalization_339/batchnorm/add/y:output:0*
T0*0
_output_shapes
:�����������
3model_87/instance_normalization_339/batchnorm/RsqrtRsqrt5model_87/instance_normalization_339/batchnorm/add:z:0*
T0*0
_output_shapes
:�����������
1model_87/instance_normalization_339/batchnorm/mulMul7model_87/instance_normalization_339/batchnorm/Rsqrt:y:04model_87/instance_normalization_339/Reshape:output:0*
T0*0
_output_shapes
:�����������
3model_87/instance_normalization_339/batchnorm/mul_1Mul$model_87/conv2d_307/BiasAdd:output:05model_87/instance_normalization_339/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
3model_87/instance_normalization_339/batchnorm/mul_2Mul9model_87/instance_normalization_339/moments/mean:output:05model_87/instance_normalization_339/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
1model_87/instance_normalization_339/batchnorm/subSub6model_87/instance_normalization_339/Reshape_1:output:07model_87/instance_normalization_339/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:�����������
3model_87/instance_normalization_339/batchnorm/add_1AddV27model_87/instance_normalization_339/batchnorm/mul_1:z:05model_87/instance_normalization_339/batchnorm/sub:z:0*
T0*0
_output_shapes
:�����������
"model_87/leaky_re_lu_383/LeakyRelu	LeakyRelu7model_87/instance_normalization_339/batchnorm/add_1:z:0*0
_output_shapes
:����������*
alpha%���>j
model_87/flatten_43/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� @  �
model_87/flatten_43/ReshapeReshape0model_87/leaky_re_lu_383/LeakyRelu:activations:0"model_87/flatten_43/Const:output:0*
T0*)
_output_shapes
:������������
'model_87/dense_43/MatMul/ReadVariableOpReadVariableOp0model_87_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model_87/dense_43/MatMulMatMul$model_87/flatten_43/Reshape:output:0/model_87/dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(model_87/dense_43/BiasAdd/ReadVariableOpReadVariableOp1model_87_dense_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_87/dense_43/BiasAddBiasAdd"model_87/dense_43/MatMul:product:00model_87/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
model_87/dense_43/SigmoidSigmoid"model_87/dense_43/BiasAdd:output:0*
T0*'
_output_shapes
:���������l
IdentityIdentitymodel_87/dense_43/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^model_87/conv2d_305/BiasAdd/ReadVariableOp*^model_87/conv2d_305/Conv2D/ReadVariableOp+^model_87/conv2d_306/BiasAdd/ReadVariableOp*^model_87/conv2d_306/Conv2D/ReadVariableOp+^model_87/conv2d_307/BiasAdd/ReadVariableOp*^model_87/conv2d_307/Conv2D/ReadVariableOp)^model_87/dense_43/BiasAdd/ReadVariableOp(^model_87/dense_43/MatMul/ReadVariableOp;^model_87/instance_normalization_338/Reshape/ReadVariableOp=^model_87/instance_normalization_338/Reshape_1/ReadVariableOp;^model_87/instance_normalization_339/Reshape/ReadVariableOp=^model_87/instance_normalization_339/Reshape_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 2X
*model_87/conv2d_305/BiasAdd/ReadVariableOp*model_87/conv2d_305/BiasAdd/ReadVariableOp2V
)model_87/conv2d_305/Conv2D/ReadVariableOp)model_87/conv2d_305/Conv2D/ReadVariableOp2X
*model_87/conv2d_306/BiasAdd/ReadVariableOp*model_87/conv2d_306/BiasAdd/ReadVariableOp2V
)model_87/conv2d_306/Conv2D/ReadVariableOp)model_87/conv2d_306/Conv2D/ReadVariableOp2X
*model_87/conv2d_307/BiasAdd/ReadVariableOp*model_87/conv2d_307/BiasAdd/ReadVariableOp2V
)model_87/conv2d_307/Conv2D/ReadVariableOp)model_87/conv2d_307/Conv2D/ReadVariableOp2T
(model_87/dense_43/BiasAdd/ReadVariableOp(model_87/dense_43/BiasAdd/ReadVariableOp2R
'model_87/dense_43/MatMul/ReadVariableOp'model_87/dense_43/MatMul/ReadVariableOp2x
:model_87/instance_normalization_338/Reshape/ReadVariableOp:model_87/instance_normalization_338/Reshape/ReadVariableOp2|
<model_87/instance_normalization_338/Reshape_1/ReadVariableOp<model_87/instance_normalization_338/Reshape_1/ReadVariableOp2x
:model_87/instance_normalization_339/Reshape/ReadVariableOp:model_87/instance_normalization_339/Reshape/ReadVariableOp2|
<model_87/instance_normalization_339/Reshape_1/ReadVariableOp<model_87/instance_normalization_339/Reshape_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
input_88
�
g
K__inference_leaky_re_lu_383_layer_call_and_return_conditional_losses_386582

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������*
alpha%���>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�i
�
__inference__traced_save_386707
file_prefixB
(read_disablecopyonread_conv2d_305_kernel:@6
(read_1_disablecopyonread_conv2d_305_bias:@E
*read_2_disablecopyonread_conv2d_306_kernel:@�7
(read_3_disablecopyonread_conv2d_306_bias:	�H
9read_4_disablecopyonread_instance_normalization_338_gamma:	�G
8read_5_disablecopyonread_instance_normalization_338_beta:	�F
*read_6_disablecopyonread_conv2d_307_kernel:��7
(read_7_disablecopyonread_conv2d_307_bias:	�H
9read_8_disablecopyonread_instance_normalization_339_gamma:	�G
8read_9_disablecopyonread_instance_normalization_339_beta:	�=
)read_10_disablecopyonread_dense_43_kernel:
��5
'read_11_disablecopyonread_dense_43_bias:
savev2_const
identity_25��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: z
Read/DisableCopyOnReadDisableCopyOnRead(read_disablecopyonread_conv2d_305_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp(read_disablecopyonread_conv2d_305_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:@|
Read_1/DisableCopyOnReadDisableCopyOnRead(read_1_disablecopyonread_conv2d_305_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp(read_1_disablecopyonread_conv2d_305_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_2/DisableCopyOnReadDisableCopyOnRead*read_2_disablecopyonread_conv2d_306_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp*read_2_disablecopyonread_conv2d_306_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0v

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�l

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�|
Read_3/DisableCopyOnReadDisableCopyOnRead(read_3_disablecopyonread_conv2d_306_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp(read_3_disablecopyonread_conv2d_306_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_4/DisableCopyOnReadDisableCopyOnRead9read_4_disablecopyonread_instance_normalization_338_gamma"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp9read_4_disablecopyonread_instance_normalization_338_gamma^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_5/DisableCopyOnReadDisableCopyOnRead8read_5_disablecopyonread_instance_normalization_338_beta"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp8read_5_disablecopyonread_instance_normalization_338_beta^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_6/DisableCopyOnReadDisableCopyOnRead*read_6_disablecopyonread_conv2d_307_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp*read_6_disablecopyonread_conv2d_307_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0x
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*(
_output_shapes
:��|
Read_7/DisableCopyOnReadDisableCopyOnRead(read_7_disablecopyonread_conv2d_307_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp(read_7_disablecopyonread_conv2d_307_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_8/DisableCopyOnReadDisableCopyOnRead9read_8_disablecopyonread_instance_normalization_339_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp9read_8_disablecopyonread_instance_normalization_339_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_9/DisableCopyOnReadDisableCopyOnRead8read_9_disablecopyonread_instance_normalization_339_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp8read_9_disablecopyonread_instance_normalization_339_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_dense_43_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_dense_43_kernel^Read_10/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_dense_43_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_dense_43_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_24Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_25IdentityIdentity_24:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_25Identity_25:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:-)
'
_user_specified_namedense_43/bias:/+
)
_user_specified_namedense_43/kernel:?
;
9
_user_specified_name!instance_normalization_339/beta:@	<
:
_user_specified_name" instance_normalization_339/gamma:/+
)
_user_specified_nameconv2d_307/bias:1-
+
_user_specified_nameconv2d_307/kernel:?;
9
_user_specified_name!instance_normalization_338/beta:@<
:
_user_specified_name" instance_normalization_338/gamma:/+
)
_user_specified_nameconv2d_306/bias:1-
+
_user_specified_nameconv2d_306/kernel:/+
)
_user_specified_nameconv2d_305/bias:1-
+
_user_specified_nameconv2d_305/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
g
K__inference_leaky_re_lu_382_layer_call_and_return_conditional_losses_386501

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������*
alpha%���>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_dense_43_layer_call_and_return_conditional_losses_386184

inputs2
matmul_readvariableop_resource:
��-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
;__inference_instance_normalization_339_layer_call_fn_386529

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *_
fZRX
V__inference_instance_normalization_339_layer_call_and_return_conditional_losses_386155x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name386525:&"
 
_user_specified_name386523:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_dense_43_layer_call_fn_386602

inputs
unknown:
��
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_386184o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name386598:&"
 
_user_specified_name386596:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�3
�
D__inference_model_87_layer_call_and_return_conditional_losses_386191
input_88+
conv2d_305_386018:@
conv2d_305_386020:@,
conv2d_306_386039:@� 
conv2d_306_386041:	�0
!instance_normalization_338_386087:	�0
!instance_normalization_338_386089:	�-
conv2d_307_386108:�� 
conv2d_307_386110:	�0
!instance_normalization_339_386156:	�0
!instance_normalization_339_386158:	�#
dense_43_386185:
��
dense_43_386187:
identity��"conv2d_305/StatefulPartitionedCall�"conv2d_306/StatefulPartitionedCall�"conv2d_307/StatefulPartitionedCall� dense_43/StatefulPartitionedCall�2instance_normalization_338/StatefulPartitionedCall�2instance_normalization_339/StatefulPartitionedCall�
"conv2d_305/StatefulPartitionedCallStatefulPartitionedCallinput_88conv2d_305_386018conv2d_305_386020*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_305_layer_call_and_return_conditional_losses_386017�
leaky_re_lu_381/PartitionedCallPartitionedCall+conv2d_305/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_381_layer_call_and_return_conditional_losses_386027�
"conv2d_306/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_381/PartitionedCall:output:0conv2d_306_386039conv2d_306_386041*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_306_layer_call_and_return_conditional_losses_386038�
2instance_normalization_338/StatefulPartitionedCallStatefulPartitionedCall+conv2d_306/StatefulPartitionedCall:output:0!instance_normalization_338_386087!instance_normalization_338_386089*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *_
fZRX
V__inference_instance_normalization_338_layer_call_and_return_conditional_losses_386086�
leaky_re_lu_382/PartitionedCallPartitionedCall;instance_normalization_338/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_382_layer_call_and_return_conditional_losses_386096�
"conv2d_307/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_382/PartitionedCall:output:0conv2d_307_386108conv2d_307_386110*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_307_layer_call_and_return_conditional_losses_386107�
2instance_normalization_339/StatefulPartitionedCallStatefulPartitionedCall+conv2d_307/StatefulPartitionedCall:output:0!instance_normalization_339_386156!instance_normalization_339_386158*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *_
fZRX
V__inference_instance_normalization_339_layer_call_and_return_conditional_losses_386155�
leaky_re_lu_383/PartitionedCallPartitionedCall;instance_normalization_339/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_383_layer_call_and_return_conditional_losses_386165�
flatten_43/PartitionedCallPartitionedCall(leaky_re_lu_383/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_43_layer_call_and_return_conditional_losses_386172�
 dense_43/StatefulPartitionedCallStatefulPartitionedCall#flatten_43/PartitionedCall:output:0dense_43_386185dense_43_386187*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_386184x
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^conv2d_305/StatefulPartitionedCall#^conv2d_306/StatefulPartitionedCall#^conv2d_307/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall3^instance_normalization_338/StatefulPartitionedCall3^instance_normalization_339/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 2H
"conv2d_305/StatefulPartitionedCall"conv2d_305/StatefulPartitionedCall2H
"conv2d_306/StatefulPartitionedCall"conv2d_306/StatefulPartitionedCall2H
"conv2d_307/StatefulPartitionedCall"conv2d_307/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2h
2instance_normalization_338/StatefulPartitionedCall2instance_normalization_338/StatefulPartitionedCall2h
2instance_normalization_339/StatefulPartitionedCall2instance_normalization_339/StatefulPartitionedCall:&"
 
_user_specified_name386187:&"
 
_user_specified_name386185:&
"
 
_user_specified_name386158:&	"
 
_user_specified_name386156:&"
 
_user_specified_name386110:&"
 
_user_specified_name386108:&"
 
_user_specified_name386089:&"
 
_user_specified_name386087:&"
 
_user_specified_name386041:&"
 
_user_specified_name386039:&"
 
_user_specified_name386020:&"
 
_user_specified_name386018:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
input_88
�
L
0__inference_leaky_re_lu_383_layer_call_fn_386577

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_383_layer_call_and_return_conditional_losses_386165i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_dense_43_layer_call_and_return_conditional_losses_386613

inputs2
matmul_readvariableop_resource:
��-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
L
0__inference_leaky_re_lu_381_layer_call_fn_386415

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_381_layer_call_and_return_conditional_losses_386027h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  @:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
G
+__inference_flatten_43_layer_call_fn_386587

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_43_layer_call_and_return_conditional_losses_386172b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_306_layer_call_fn_386429

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_306_layer_call_and_return_conditional_losses_386038x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name386425:&"
 
_user_specified_name386423:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_383_layer_call_and_return_conditional_losses_386165

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������*
alpha%���>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_model_87_layer_call_fn_386287
input_88!
unknown:@
	unknown_0:@$
	unknown_1:@�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_88unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_87_layer_call_and_return_conditional_losses_386229o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name386283:&"
 
_user_specified_name386281:&
"
 
_user_specified_name386279:&	"
 
_user_specified_name386277:&"
 
_user_specified_name386275:&"
 
_user_specified_name386273:&"
 
_user_specified_name386271:&"
 
_user_specified_name386269:&"
 
_user_specified_name386267:&"
 
_user_specified_name386265:&"
 
_user_specified_name386263:&"
 
_user_specified_name386261:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
input_88
�
�
+__inference_conv2d_307_layer_call_fn_386510

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_307_layer_call_and_return_conditional_losses_386107x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name386506:&"
 
_user_specified_name386504:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_conv2d_306_layer_call_and_return_conditional_losses_386439

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�3
�
D__inference_model_87_layer_call_and_return_conditional_losses_386229
input_88+
conv2d_305_386194:@
conv2d_305_386196:@,
conv2d_306_386200:@� 
conv2d_306_386202:	�0
!instance_normalization_338_386205:	�0
!instance_normalization_338_386207:	�-
conv2d_307_386211:�� 
conv2d_307_386213:	�0
!instance_normalization_339_386216:	�0
!instance_normalization_339_386218:	�#
dense_43_386223:
��
dense_43_386225:
identity��"conv2d_305/StatefulPartitionedCall�"conv2d_306/StatefulPartitionedCall�"conv2d_307/StatefulPartitionedCall� dense_43/StatefulPartitionedCall�2instance_normalization_338/StatefulPartitionedCall�2instance_normalization_339/StatefulPartitionedCall�
"conv2d_305/StatefulPartitionedCallStatefulPartitionedCallinput_88conv2d_305_386194conv2d_305_386196*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_305_layer_call_and_return_conditional_losses_386017�
leaky_re_lu_381/PartitionedCallPartitionedCall+conv2d_305/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_381_layer_call_and_return_conditional_losses_386027�
"conv2d_306/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_381/PartitionedCall:output:0conv2d_306_386200conv2d_306_386202*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_306_layer_call_and_return_conditional_losses_386038�
2instance_normalization_338/StatefulPartitionedCallStatefulPartitionedCall+conv2d_306/StatefulPartitionedCall:output:0!instance_normalization_338_386205!instance_normalization_338_386207*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *_
fZRX
V__inference_instance_normalization_338_layer_call_and_return_conditional_losses_386086�
leaky_re_lu_382/PartitionedCallPartitionedCall;instance_normalization_338/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_382_layer_call_and_return_conditional_losses_386096�
"conv2d_307/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_382/PartitionedCall:output:0conv2d_307_386211conv2d_307_386213*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_307_layer_call_and_return_conditional_losses_386107�
2instance_normalization_339/StatefulPartitionedCallStatefulPartitionedCall+conv2d_307/StatefulPartitionedCall:output:0!instance_normalization_339_386216!instance_normalization_339_386218*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *_
fZRX
V__inference_instance_normalization_339_layer_call_and_return_conditional_losses_386155�
leaky_re_lu_383/PartitionedCallPartitionedCall;instance_normalization_339/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_383_layer_call_and_return_conditional_losses_386165�
flatten_43/PartitionedCallPartitionedCall(leaky_re_lu_383/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_43_layer_call_and_return_conditional_losses_386172�
 dense_43/StatefulPartitionedCallStatefulPartitionedCall#flatten_43/PartitionedCall:output:0dense_43_386223dense_43_386225*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_386184x
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^conv2d_305/StatefulPartitionedCall#^conv2d_306/StatefulPartitionedCall#^conv2d_307/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall3^instance_normalization_338/StatefulPartitionedCall3^instance_normalization_339/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 2H
"conv2d_305/StatefulPartitionedCall"conv2d_305/StatefulPartitionedCall2H
"conv2d_306/StatefulPartitionedCall"conv2d_306/StatefulPartitionedCall2H
"conv2d_307/StatefulPartitionedCall"conv2d_307/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2h
2instance_normalization_338/StatefulPartitionedCall2instance_normalization_338/StatefulPartitionedCall2h
2instance_normalization_339/StatefulPartitionedCall2instance_normalization_339/StatefulPartitionedCall:&"
 
_user_specified_name386225:&"
 
_user_specified_name386223:&
"
 
_user_specified_name386218:&	"
 
_user_specified_name386216:&"
 
_user_specified_name386213:&"
 
_user_specified_name386211:&"
 
_user_specified_name386207:&"
 
_user_specified_name386205:&"
 
_user_specified_name386202:&"
 
_user_specified_name386200:&"
 
_user_specified_name386196:&"
 
_user_specified_name386194:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
input_88
�
g
K__inference_leaky_re_lu_381_layer_call_and_return_conditional_losses_386027

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������  @*
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  @:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�

�
F__inference_conv2d_307_layer_call_and_return_conditional_losses_386107

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
0__inference_leaky_re_lu_382_layer_call_fn_386496

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_382_layer_call_and_return_conditional_losses_386096i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_305_layer_call_fn_386400

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_305_layer_call_and_return_conditional_losses_386017w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name386396:&"
 
_user_specified_name386394:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
b
F__inference_flatten_43_layer_call_and_return_conditional_losses_386172

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� @  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_conv2d_307_layer_call_and_return_conditional_losses_386520

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_conv2d_305_layer_call_and_return_conditional_losses_386410

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������  @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�*
�
V__inference_instance_normalization_338_layer_call_and_return_conditional_losses_386491

inputs.
reshape_readvariableop_resource:	�0
!reshape_1_readvariableop_resource:	�
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(v
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:�����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:����������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(s
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:�*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:�w
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:�T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*0
_output_shapes
:����������f
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*0
_output_shapes
:����������v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:����������l
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*0
_output_shapes
:����������{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:����������x
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:����������{
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*0
_output_shapes
:����������k
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*0
_output_shapes
:����������V
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�*
�
V__inference_instance_normalization_338_layer_call_and_return_conditional_losses_386086

inputs.
reshape_readvariableop_resource:	�0
!reshape_1_readvariableop_resource:	�
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(v
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:�����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:����������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(s
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:�*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:�w
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:�T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*0
_output_shapes
:����������f
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*0
_output_shapes
:����������v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:����������l
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*0
_output_shapes
:����������{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:����������x
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:����������{
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*0
_output_shapes
:����������k
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*0
_output_shapes
:����������V
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�=
�
"__inference__traced_restore_386752
file_prefix<
"assignvariableop_conv2d_305_kernel:@0
"assignvariableop_1_conv2d_305_bias:@?
$assignvariableop_2_conv2d_306_kernel:@�1
"assignvariableop_3_conv2d_306_bias:	�B
3assignvariableop_4_instance_normalization_338_gamma:	�A
2assignvariableop_5_instance_normalization_338_beta:	�@
$assignvariableop_6_conv2d_307_kernel:��1
"assignvariableop_7_conv2d_307_bias:	�B
3assignvariableop_8_instance_normalization_339_gamma:	�A
2assignvariableop_9_instance_normalization_339_beta:	�7
#assignvariableop_10_dense_43_kernel:
��/
!assignvariableop_11_dense_43_bias:
identity_13��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_305_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_305_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_306_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_306_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp3assignvariableop_4_instance_normalization_338_gammaIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp2assignvariableop_5_instance_normalization_338_betaIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp$assignvariableop_6_conv2d_307_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_307_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp3assignvariableop_8_instance_normalization_339_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp2assignvariableop_9_instance_normalization_339_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_43_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_43_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_13Identity_13:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:-)
'
_user_specified_namedense_43/bias:/+
)
_user_specified_namedense_43/kernel:?
;
9
_user_specified_name!instance_normalization_339/beta:@	<
:
_user_specified_name" instance_normalization_339/gamma:/+
)
_user_specified_nameconv2d_307/bias:1-
+
_user_specified_nameconv2d_307/kernel:?;
9
_user_specified_name!instance_normalization_338/beta:@<
:
_user_specified_name" instance_normalization_338/gamma:/+
)
_user_specified_nameconv2d_306/bias:1-
+
_user_specified_nameconv2d_306/kernel:/+
)
_user_specified_nameconv2d_305/bias:1-
+
_user_specified_nameconv2d_305/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
)__inference_model_87_layer_call_fn_386258
input_88!
unknown:@
	unknown_0:@$
	unknown_1:@�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_88unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_87_layer_call_and_return_conditional_losses_386191o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name386254:&"
 
_user_specified_name386252:&
"
 
_user_specified_name386250:&	"
 
_user_specified_name386248:&"
 
_user_specified_name386246:&"
 
_user_specified_name386244:&"
 
_user_specified_name386242:&"
 
_user_specified_name386240:&"
 
_user_specified_name386238:&"
 
_user_specified_name386236:&"
 
_user_specified_name386234:&"
 
_user_specified_name386232:Y U
/
_output_shapes
:���������@@
"
_user_specified_name
input_88
�
g
K__inference_leaky_re_lu_382_layer_call_and_return_conditional_losses_386096

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������*
alpha%���>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_conv2d_305_layer_call_and_return_conditional_losses_386017

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������  @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_381_layer_call_and_return_conditional_losses_386420

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������  @*
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  @:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
b
F__inference_flatten_43_layer_call_and_return_conditional_losses_386593

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� @  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�*
�
V__inference_instance_normalization_339_layer_call_and_return_conditional_losses_386572

inputs.
reshape_readvariableop_resource:	�0
!reshape_1_readvariableop_resource:	�
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(v
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:�����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:����������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(s
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:�*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:�w
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:�T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*0
_output_shapes
:����������f
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*0
_output_shapes
:����������v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:����������l
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*0
_output_shapes
:����������{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:����������x
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:����������{
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*0
_output_shapes
:����������k
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*0
_output_shapes
:����������V
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_conv2d_306_layer_call_and_return_conditional_losses_386038

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_889
serving_default_input_88:0���������@@<
dense_430
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer-9
layer_with_weights-5
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias
 +_jit_compiled_convolution_op"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
	2gamma
3beta"
_tf_keras_layer
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias
 B_jit_compiled_convolution_op"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
	Igamma
Jbeta"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

]kernel
^bias"
_tf_keras_layer
v
0
1
)2
*3
24
35
@6
A7
I8
J9
]10
^11"
trackable_list_wrapper
v
0
1
)2
*3
24
35
@6
A7
I8
J9
]10
^11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
dtrace_0
etrace_12�
)__inference_model_87_layer_call_fn_386258
)__inference_model_87_layer_call_fn_386287�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0zetrace_1
�
ftrace_0
gtrace_12�
D__inference_model_87_layer_call_and_return_conditional_losses_386191
D__inference_model_87_layer_call_and_return_conditional_losses_386229�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zftrace_0zgtrace_1
�B�
!__inference__wrapped_model_386005input_88"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
hserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
ntrace_02�
+__inference_conv2d_305_layer_call_fn_386400�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zntrace_0
�
otrace_02�
F__inference_conv2d_305_layer_call_and_return_conditional_losses_386410�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zotrace_0
+:)@2conv2d_305/kernel
:@2conv2d_305/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
utrace_02�
0__inference_leaky_re_lu_381_layer_call_fn_386415�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zutrace_0
�
vtrace_02�
K__inference_leaky_re_lu_381_layer_call_and_return_conditional_losses_386420�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zvtrace_0
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
|trace_02�
+__inference_conv2d_306_layer_call_fn_386429�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z|trace_0
�
}trace_02�
F__inference_conv2d_306_layer_call_and_return_conditional_losses_386439�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z}trace_0
,:*@�2conv2d_306/kernel
:�2conv2d_306/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
;__inference_instance_normalization_338_layer_call_fn_386448�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
V__inference_instance_normalization_338_layer_call_and_return_conditional_losses_386491�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:-�2 instance_normalization_338/gamma
.:,�2instance_normalization_338/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_leaky_re_lu_382_layer_call_fn_386496�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_leaky_re_lu_382_layer_call_and_return_conditional_losses_386501�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_conv2d_307_layer_call_fn_386510�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_conv2d_307_layer_call_and_return_conditional_losses_386520�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+��2conv2d_307/kernel
:�2conv2d_307/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
;__inference_instance_normalization_339_layer_call_fn_386529�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
V__inference_instance_normalization_339_layer_call_and_return_conditional_losses_386572�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:-�2 instance_normalization_339/gamma
.:,�2instance_normalization_339/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_leaky_re_lu_383_layer_call_fn_386577�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_leaky_re_lu_383_layer_call_and_return_conditional_losses_386582�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_flatten_43_layer_call_fn_386587�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_flatten_43_layer_call_and_return_conditional_losses_386593�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_43_layer_call_fn_386602�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_43_layer_call_and_return_conditional_losses_386613�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!
��2dense_43/kernel
:2dense_43/bias
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_model_87_layer_call_fn_386258input_88"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_87_layer_call_fn_386287input_88"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_87_layer_call_and_return_conditional_losses_386191input_88"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_87_layer_call_and_return_conditional_losses_386229input_88"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_386391input_88"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_conv2d_305_layer_call_fn_386400inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_conv2d_305_layer_call_and_return_conditional_losses_386410inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_leaky_re_lu_381_layer_call_fn_386415inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_leaky_re_lu_381_layer_call_and_return_conditional_losses_386420inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_conv2d_306_layer_call_fn_386429inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_conv2d_306_layer_call_and_return_conditional_losses_386439inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
;__inference_instance_normalization_338_layer_call_fn_386448inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
V__inference_instance_normalization_338_layer_call_and_return_conditional_losses_386491inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_leaky_re_lu_382_layer_call_fn_386496inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_leaky_re_lu_382_layer_call_and_return_conditional_losses_386501inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_conv2d_307_layer_call_fn_386510inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_conv2d_307_layer_call_and_return_conditional_losses_386520inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
;__inference_instance_normalization_339_layer_call_fn_386529inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
V__inference_instance_normalization_339_layer_call_and_return_conditional_losses_386572inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_leaky_re_lu_383_layer_call_fn_386577inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_leaky_re_lu_383_layer_call_and_return_conditional_losses_386582inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_flatten_43_layer_call_fn_386587inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_flatten_43_layer_call_and_return_conditional_losses_386593inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_43_layer_call_fn_386602inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_43_layer_call_and_return_conditional_losses_386613inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_386005~)*23@AIJ]^9�6
/�,
*�'
input_88���������@@
� "3�0
.
dense_43"�
dense_43����������
F__inference_conv2d_305_layer_call_and_return_conditional_losses_386410s7�4
-�*
(�%
inputs���������@@
� "4�1
*�'
tensor_0���������  @
� �
+__inference_conv2d_305_layer_call_fn_386400h7�4
-�*
(�%
inputs���������@@
� ")�&
unknown���������  @�
F__inference_conv2d_306_layer_call_and_return_conditional_losses_386439t)*7�4
-�*
(�%
inputs���������  @
� "5�2
+�(
tensor_0����������
� �
+__inference_conv2d_306_layer_call_fn_386429i)*7�4
-�*
(�%
inputs���������  @
� "*�'
unknown�����������
F__inference_conv2d_307_layer_call_and_return_conditional_losses_386520u@A8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
+__inference_conv2d_307_layer_call_fn_386510j@A8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
D__inference_dense_43_layer_call_and_return_conditional_losses_386613e]^1�.
'�$
"�
inputs�����������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_43_layer_call_fn_386602Z]^1�.
'�$
"�
inputs�����������
� "!�
unknown����������
F__inference_flatten_43_layer_call_and_return_conditional_losses_386593j8�5
.�+
)�&
inputs����������
� ".�+
$�!
tensor_0�����������
� �
+__inference_flatten_43_layer_call_fn_386587_8�5
.�+
)�&
inputs����������
� "#� 
unknown������������
V__inference_instance_normalization_338_layer_call_and_return_conditional_losses_386491u238�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
;__inference_instance_normalization_338_layer_call_fn_386448j238�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
V__inference_instance_normalization_339_layer_call_and_return_conditional_losses_386572uIJ8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
;__inference_instance_normalization_339_layer_call_fn_386529jIJ8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
K__inference_leaky_re_lu_381_layer_call_and_return_conditional_losses_386420o7�4
-�*
(�%
inputs���������  @
� "4�1
*�'
tensor_0���������  @
� �
0__inference_leaky_re_lu_381_layer_call_fn_386415d7�4
-�*
(�%
inputs���������  @
� ")�&
unknown���������  @�
K__inference_leaky_re_lu_382_layer_call_and_return_conditional_losses_386501q8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
0__inference_leaky_re_lu_382_layer_call_fn_386496f8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
K__inference_leaky_re_lu_383_layer_call_and_return_conditional_losses_386582q8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
0__inference_leaky_re_lu_383_layer_call_fn_386577f8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
D__inference_model_87_layer_call_and_return_conditional_losses_386191)*23@AIJ]^A�>
7�4
*�'
input_88���������@@
p

 
� ",�)
"�
tensor_0���������
� �
D__inference_model_87_layer_call_and_return_conditional_losses_386229)*23@AIJ]^A�>
7�4
*�'
input_88���������@@
p 

 
� ",�)
"�
tensor_0���������
� �
)__inference_model_87_layer_call_fn_386258t)*23@AIJ]^A�>
7�4
*�'
input_88���������@@
p

 
� "!�
unknown����������
)__inference_model_87_layer_call_fn_386287t)*23@AIJ]^A�>
7�4
*�'
input_88���������@@
p 

 
� "!�
unknown����������
$__inference_signature_wrapper_386391�)*23@AIJ]^E�B
� 
;�8
6
input_88*�'
input_88���������@@"3�0
.
dense_43"�
dense_43���������