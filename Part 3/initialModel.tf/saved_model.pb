ƈ
?'?'
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameembedding_2/embeddings
?
*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings*
_output_shapes
:	?*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
m

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name45974*
value_dtype0	
?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_37314*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/embedding_2/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*.
shared_nameAdam/embedding_2/embeddings/m
?
1Adam/embedding_2/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_2/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*.
shared_nameAdam/embedding_2/embeddings/v
?
1Adam/embedding_2/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?

Const_4Const*
_output_shapes	
:?*
dtype0*?	
value?	B?	?BjhinBluxBcaitlynBmissfortuneBezrealByasuoBvayneBjinxBgravesByoneBasheBkaynBlucianBekkoBsettBmorganaBviegoBmasteryiByuumiBireliaBteemoBnamiBtristanaBwarwickBthreshBzedBleesinBsennaBleonaBpykeBluluBjaxBkaisaBgarenBkhazixBcamilleBxerathBsorakaB
blitzcrankBbrandBtwitchBdravenBshacoB
tryndamereBmordekaiserBsylasBviBdariusBnautilusBveigarBakaliBnasusBvexBxinzhaoBviktorBtalonBkatarinaBsamiraBamumuBnocturneBkayleBsonaBvolibearBzyraBakshanBleblancBhecarimBtrundleBmalphiteBdianaBapheliosBdrmundoBfioraBshenB	seraphineBxayahBvladimirBahriB	tahmkenchBrivenBjayceBsionBkarmaBziggsBqiyanaBpantheonBswainBurgotBaatroxBnunuBjarvanivBlilliaBfizzBkindredBannieBchogathBevelynnBsivirBkassadinBmalzaharBtwistedfateB
monkeykingBfiddlesticksBgalioBzacBrengarBjannaBrakanBzileanBaniviaB	gangplankBgwenBbardBryzeBpoppyByorickBvelkozBalistarBvarusBheimerdingerBsingedBsyndraBzoeBrenektonBillaoiBkarthusBnidaleeBkennenBkogmawBmaokaiBshyvanaBoriannaB
cassiopeiaBneekoBornnBbraumB	lissandraBudyrBrammusBkledBeliseBcorkiBgragasBrellBreksaiBsejuaniBgnarBquinnBtaliyahBolafBrumbleBtaricBkalistaBivernBazirBskarnerBaurelionsol
?

Const_5Const*
_output_shapes	
:?*
dtype0	*?

value?	B?		?"?	                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_57622
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_57627
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?+
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?*
value?*B?* B?*
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
 
"
_lookup_layer
	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
R
%	variables
&trainable_variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?
/iter

0beta_1

1beta_2
	2decay
3learning_ratemfmg mh)mi*mjvkvl vm)vn*vo
#
1
2
 3
)4
*5
#
0
1
 2
)3
*4
 
?
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
 
3
9lookup_table
:token_counts
;	keras_api
 
fd
VARIABLE_VALUEembedding_2/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
!	variables
"trainable_variables
#regularization_losses
 
 
 
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
%	variables
&trainable_variables
'regularization_losses
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
+	variables
,trainable_variables
-regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8

Z0
[1
 
 

\_initializer
LJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	]total
	^count
_	variables
`	keras_api
D
	atotal
	bcount
c
_fn_kwargs
d	variables
e	keras_api
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

]0
^1

_	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

a0
b1

d	variables
??
VARIABLE_VALUEAdam/embedding_2/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_2/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_5Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_6Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_input_5serving_default_input_6
hash_tableConstConst_1Const_2embedding_2/embeddingsdense_5/kerneldense_5/biasdense_4/kerneldense_4/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_57076
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename*embedding_2/embeddings/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp1Adam/embedding_2/embeddings/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp1Adam/embedding_2/embeddings/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOpConst_6*'
Tin 
2		*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_57737
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding_2/embeddingsdense_5/kerneldense_5/biasdense_4/kerneldense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotalcounttotal_1count_1Adam/embedding_2/embeddings/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/embedding_2/embeddings/vAdam/dense_5/kernel/vAdam/dense_5/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/v*%
Tin
2*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_57822??
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_57515

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__initializer_575608
4key_value_init45973_lookuptableimportv2_table_handle0
,key_value_init45973_lookuptableimportv2_keys2
.key_value_init45973_lookuptableimportv2_values	
identity??'key_value_init45973/LookupTableImportV2?
'key_value_init45973/LookupTableImportV2LookupTableImportV24key_value_init45973_lookuptableimportv2_table_handle,key_value_init45973_lookuptableimportv2_keys.key_value_init45973_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init45973/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2R
'key_value_init45973/LookupTableImportV2'key_value_init45973/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
'__inference_model_2_layer_call_fn_57100
inputs_0
inputs_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_56543o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?C
?
__inference_adapt_step_57429
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?
.
__inference__initializer_57575
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
F__inference_embedding_2_layer_call_and_return_conditional_losses_57445

inputs	)
embedding_lookup_57439:	?
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_57439inputs*
Tindices0	*)
_class
loc:@embedding_lookup/57439*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/57439*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?g
?
!__inference__traced_restore_57822
file_prefix:
'assignvariableop_embedding_2_embeddings:	?3
!assignvariableop_1_dense_5_kernel:-
assignvariableop_2_dense_5_bias:3
!assignvariableop_3_dense_4_kernel:-
assignvariableop_4_dense_4_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: #
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: D
1assignvariableop_14_adam_embedding_2_embeddings_m:	?;
)assignvariableop_15_adam_dense_5_kernel_m:5
'assignvariableop_16_adam_dense_5_bias_m:;
)assignvariableop_17_adam_dense_4_kernel_m:5
'assignvariableop_18_adam_dense_4_bias_m:D
1assignvariableop_19_adam_embedding_2_embeddings_v:	?;
)assignvariableop_20_adam_dense_5_kernel_v:5
'assignvariableop_21_adam_dense_5_bias_v:;
)assignvariableop_22_adam_dense_4_kernel_v:5
'assignvariableop_23_adam_dense_4_bias_v:
identity_25??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_2_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_5_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_5_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_4_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_4_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:10RestoreV2:tensors:11*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 _
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp1assignvariableop_14_adam_embedding_2_embeddings_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_5_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_5_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_4_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp1assignvariableop_19_adam_embedding_2_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_5_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_5_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_4_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_4_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
?
?
__inference_save_fn_57599
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
'__inference_model_2_layer_call_fn_56814
input_5
input_6
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_56769o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
,
__inference__destroyer_57580
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_56929
input_5
input_6S
Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_2_string_lookup_2_equal_y3
/text_vectorization_2_string_lookup_2_selectv2_t	$
embedding_2_56910:	?
dense_5_56917:
dense_5_56919:
dense_4_56923:
dense_4_56925:
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?%embedding_2/StatefulPartitionedCall_1?Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2?Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2a
 text_vectorization_2/StringLowerStringLowerinput_6*'
_output_shapes
:??????????
'text_vectorization_2/StaticRegexReplaceStaticRegexReplace)text_vectorization_2/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_2/SqueezeSqueeze0text_vectorization_2/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_2/StringSplit/StringSplitV2StringSplitV2%text_vectorization_2/Squeeze:output:0/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_2/StringSplit/strided_sliceStridedSlice8text_vectorization_2/StringSplit/StringSplitV2:indices:0=text_vectorization_2/StringSplit/strided_slice/stack:output:0?text_vectorization_2/StringSplit/strided_slice/stack_1:output:0?text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_2/StringSplit/strided_slice_1StridedSlice6text_vectorization_2/StringSplit/StringSplitV2:shape:0?text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle7text_vectorization_2/StringSplit/StringSplitV2:values:0Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_2/string_lookup_2/EqualEqual7text_vectorization_2/StringSplit/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/SelectV2SelectV2.text_vectorization_2/string_lookup_2/Equal:z:0/text_vectorization_2_string_lookup_2_selectv2_tKtext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/IdentityIdentity6text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
8text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_2/RaggedToTensor/Const:output:06text_vectorization_2/string_lookup_2/Identity:output:0:text_vectorization_2/RaggedToTensor/default_value:output:09text_vectorization_2/StringSplit/strided_slice_1:output:07text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSc
"text_vectorization_2/StringLower_1StringLowerinput_5*'
_output_shapes
:??????????
)text_vectorization_2/StaticRegexReplace_1StaticRegexReplace+text_vectorization_2/StringLower_1:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_2/Squeeze_1Squeeze2text_vectorization_2/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????i
(text_vectorization_2/StringSplit_1/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
0text_vectorization_2/StringSplit_1/StringSplitV2StringSplitV2'text_vectorization_2/Squeeze_1:output:01text_vectorization_2/StringSplit_1/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
6text_vectorization_2/StringSplit_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
8text_vectorization_2/StringSplit_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
8text_vectorization_2/StringSplit_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
0text_vectorization_2/StringSplit_1/strided_sliceStridedSlice:text_vectorization_2/StringSplit_1/StringSplitV2:indices:0?text_vectorization_2/StringSplit_1/strided_slice/stack:output:0Atext_vectorization_2/StringSplit_1/strided_slice/stack_1:output:0Atext_vectorization_2/StringSplit_1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
8text_vectorization_2/StringSplit_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:text_vectorization_2/StringSplit_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:text_vectorization_2/StringSplit_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2text_vectorization_2/StringSplit_1/strided_slice_1StridedSlice8text_vectorization_2/StringSplit_1/StringSplitV2:shape:0Atext_vectorization_2/StringSplit_1/strided_slice_1/stack:output:0Ctext_vectorization_2/StringSplit_1/strided_slice_1/stack_1:output:0Ctext_vectorization_2/StringSplit_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Ytext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast9text_vectorization_2/StringSplit_1/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast;text_vectorization_2/StringSplit_1/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
btext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
gtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterktext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ptext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
btext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastitext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ntext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2jtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulftext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum_text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum_text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
ftext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ntext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsummtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
dtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
`text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2mtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2LookupTableFindV2Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle9text_vectorization_2/StringSplit_1/StringSplitV2:values:0Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_valueC^text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,text_vectorization_2/string_lookup_2/Equal_1Equal9text_vectorization_2/StringSplit_1/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
/text_vectorization_2/string_lookup_2/SelectV2_1SelectV20text_vectorization_2/string_lookup_2/Equal_1:z:0/text_vectorization_2_string_lookup_2_selectv2_tMtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/text_vectorization_2/string_lookup_2/Identity_1Identity8text_vectorization_2/string_lookup_2/SelectV2_1:output:0*
T0	*#
_output_shapes
:?????????u
3text_vectorization_2/RaggedToTensor_1/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
+text_vectorization_2/RaggedToTensor_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
:text_vectorization_2/RaggedToTensor_1/RaggedTensorToTensorRaggedTensorToTensor4text_vectorization_2/RaggedToTensor_1/Const:output:08text_vectorization_2/string_lookup_2/Identity_1:output:0<text_vectorization_2/RaggedToTensor_1/default_value:output:0;text_vectorization_2/StringSplit_1/strided_slice_1:output:09text_vectorization_2/StringSplit_1/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallCtext_vectorization_2/RaggedToTensor_1/RaggedTensorToTensor:result:0embedding_2_56910*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_56479?
%embedding_2/StatefulPartitionedCall_1StatefulPartitionedCallAtext_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0embedding_2_56910*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_56479?
concatenate_2/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0.embedding_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_56492?
*global_average_pooling1d_2/PartitionedCallPartitionedCall&concatenate_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_56499?
dense_5/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0dense_5_56917dense_5_56919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_56512?
dropout_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_56523?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_4_56923dense_4_56925*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_56536w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall&^embedding_2/StatefulPartitionedCall_1C^text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2E^text_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????:?????????: : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2N
%embedding_2/StatefulPartitionedCall_1%embedding_2/StatefulPartitionedCall_12?
Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV22?
Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
V
:__inference_global_average_pooling1d_2_layer_call_fn_57468

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_56499`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_56356
input_5
input_6[
Wmodel_2_text_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle\
Xmodel_2_text_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value	8
4model_2_text_vectorization_2_string_lookup_2_equal_y;
7model_2_text_vectorization_2_string_lookup_2_selectv2_t	=
*model_2_embedding_2_embedding_lookup_56327:	?@
.model_2_dense_5_matmul_readvariableop_resource:=
/model_2_dense_5_biasadd_readvariableop_resource:@
.model_2_dense_4_matmul_readvariableop_resource:=
/model_2_dense_4_biasadd_readvariableop_resource:
identity??&model_2/dense_4/BiasAdd/ReadVariableOp?%model_2/dense_4/MatMul/ReadVariableOp?&model_2/dense_5/BiasAdd/ReadVariableOp?%model_2/dense_5/MatMul/ReadVariableOp?$model_2/embedding_2/embedding_lookup?&model_2/embedding_2/embedding_lookup_1?Jmodel_2/text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2?Lmodel_2/text_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2i
(model_2/text_vectorization_2/StringLowerStringLowerinput_6*'
_output_shapes
:??????????
/model_2/text_vectorization_2/StaticRegexReplaceStaticRegexReplace1model_2/text_vectorization_2/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
$model_2/text_vectorization_2/SqueezeSqueeze8model_2/text_vectorization_2/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????o
.model_2/text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
6model_2/text_vectorization_2/StringSplit/StringSplitV2StringSplitV2-model_2/text_vectorization_2/Squeeze:output:07model_2/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
<model_2/text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
>model_2/text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
>model_2/text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
6model_2/text_vectorization_2/StringSplit/strided_sliceStridedSlice@model_2/text_vectorization_2/StringSplit/StringSplitV2:indices:0Emodel_2/text_vectorization_2/StringSplit/strided_slice/stack:output:0Gmodel_2/text_vectorization_2/StringSplit/strided_slice/stack_1:output:0Gmodel_2/text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
>model_2/text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@model_2/text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@model_2/text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8model_2/text_vectorization_2/StringSplit/strided_slice_1StridedSlice>model_2/text_vectorization_2/StringSplit/StringSplitV2:shape:0Gmodel_2/text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Imodel_2/text_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Imodel_2/text_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
_model_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast?model_2/text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
amodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastAmodel_2/text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
imodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapecmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
imodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
hmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdrmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0rmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
mmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
kmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterqmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0vmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
hmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastomodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
kmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
gmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxcmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0tmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
imodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
gmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2pmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0rmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
gmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMullmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0kmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
kmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumemodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
kmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumemodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0omodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
kmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
lmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountcmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0omodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0tmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
fmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
amodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumsmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0omodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
jmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
fmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
amodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2smodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0gmodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0omodel_2/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Jmodel_2/text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2Wmodel_2_text_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle?model_2/text_vectorization_2/StringSplit/StringSplitV2:values:0Xmodel_2_text_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
2model_2/text_vectorization_2/string_lookup_2/EqualEqual?model_2/text_vectorization_2/StringSplit/StringSplitV2:values:04model_2_text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
5model_2/text_vectorization_2/string_lookup_2/SelectV2SelectV26model_2/text_vectorization_2/string_lookup_2/Equal:z:07model_2_text_vectorization_2_string_lookup_2_selectv2_tSmodel_2/text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
5model_2/text_vectorization_2/string_lookup_2/IdentityIdentity>model_2/text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????{
9model_2/text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
1model_2/text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
@model_2/text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor:model_2/text_vectorization_2/RaggedToTensor/Const:output:0>model_2/text_vectorization_2/string_lookup_2/Identity:output:0Bmodel_2/text_vectorization_2/RaggedToTensor/default_value:output:0Amodel_2/text_vectorization_2/StringSplit/strided_slice_1:output:0?model_2/text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSk
*model_2/text_vectorization_2/StringLower_1StringLowerinput_5*'
_output_shapes
:??????????
1model_2/text_vectorization_2/StaticRegexReplace_1StaticRegexReplace3model_2/text_vectorization_2/StringLower_1:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
&model_2/text_vectorization_2/Squeeze_1Squeeze:model_2/text_vectorization_2/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????q
0model_2/text_vectorization_2/StringSplit_1/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
8model_2/text_vectorization_2/StringSplit_1/StringSplitV2StringSplitV2/model_2/text_vectorization_2/Squeeze_1:output:09model_2/text_vectorization_2/StringSplit_1/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
>model_2/text_vectorization_2/StringSplit_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
@model_2/text_vectorization_2/StringSplit_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
@model_2/text_vectorization_2/StringSplit_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
8model_2/text_vectorization_2/StringSplit_1/strided_sliceStridedSliceBmodel_2/text_vectorization_2/StringSplit_1/StringSplitV2:indices:0Gmodel_2/text_vectorization_2/StringSplit_1/strided_slice/stack:output:0Imodel_2/text_vectorization_2/StringSplit_1/strided_slice/stack_1:output:0Imodel_2/text_vectorization_2/StringSplit_1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
@model_2/text_vectorization_2/StringSplit_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bmodel_2/text_vectorization_2/StringSplit_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bmodel_2/text_vectorization_2/StringSplit_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:model_2/text_vectorization_2/StringSplit_1/strided_slice_1StridedSlice@model_2/text_vectorization_2/StringSplit_1/StringSplitV2:shape:0Imodel_2/text_vectorization_2/StringSplit_1/strided_slice_1/stack:output:0Kmodel_2/text_vectorization_2/StringSplit_1/strided_slice_1/stack_1:output:0Kmodel_2/text_vectorization_2/StringSplit_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
amodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastAmodel_2/text_vectorization_2/StringSplit_1/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
cmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastCmodel_2/text_vectorization_2/StringSplit_1/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
kmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeemodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
kmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
jmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdtmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0tmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
omodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
mmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatersmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0xmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
jmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastqmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
mmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
imodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxemodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0vmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
kmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
imodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2rmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0tmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
imodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulnmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0mmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
mmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumgmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0mmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
mmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumgmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0qmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
mmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
nmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountemodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0qmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0vmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
hmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
cmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumumodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0qmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
lmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
hmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
cmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2umodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0imodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0qmodel_2/text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Lmodel_2/text_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2LookupTableFindV2Wmodel_2_text_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handleAmodel_2/text_vectorization_2/StringSplit_1/StringSplitV2:values:0Xmodel_2_text_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_valueK^model_2/text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*#
_output_shapes
:??????????
4model_2/text_vectorization_2/string_lookup_2/Equal_1EqualAmodel_2/text_vectorization_2/StringSplit_1/StringSplitV2:values:04model_2_text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
7model_2/text_vectorization_2/string_lookup_2/SelectV2_1SelectV28model_2/text_vectorization_2/string_lookup_2/Equal_1:z:07model_2_text_vectorization_2_string_lookup_2_selectv2_tUmodel_2/text_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
7model_2/text_vectorization_2/string_lookup_2/Identity_1Identity@model_2/text_vectorization_2/string_lookup_2/SelectV2_1:output:0*
T0	*#
_output_shapes
:?????????}
;model_2/text_vectorization_2/RaggedToTensor_1/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
3model_2/text_vectorization_2/RaggedToTensor_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
Bmodel_2/text_vectorization_2/RaggedToTensor_1/RaggedTensorToTensorRaggedTensorToTensor<model_2/text_vectorization_2/RaggedToTensor_1/Const:output:0@model_2/text_vectorization_2/string_lookup_2/Identity_1:output:0Dmodel_2/text_vectorization_2/RaggedToTensor_1/default_value:output:0Cmodel_2/text_vectorization_2/StringSplit_1/strided_slice_1:output:0Amodel_2/text_vectorization_2/StringSplit_1/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
$model_2/embedding_2/embedding_lookupResourceGather*model_2_embedding_2_embedding_lookup_56327Kmodel_2/text_vectorization_2/RaggedToTensor_1/RaggedTensorToTensor:result:0*
Tindices0	*=
_class3
1/loc:@model_2/embedding_2/embedding_lookup/56327*+
_output_shapes
:?????????*
dtype0?
-model_2/embedding_2/embedding_lookup/IdentityIdentity-model_2/embedding_2/embedding_lookup:output:0*
T0*=
_class3
1/loc:@model_2/embedding_2/embedding_lookup/56327*+
_output_shapes
:??????????
/model_2/embedding_2/embedding_lookup/Identity_1Identity6model_2/embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
&model_2/embedding_2/embedding_lookup_1ResourceGather*model_2_embedding_2_embedding_lookup_56327Imodel_2/text_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*=
_class3
1/loc:@model_2/embedding_2/embedding_lookup/56327*+
_output_shapes
:?????????*
dtype0?
/model_2/embedding_2/embedding_lookup_1/IdentityIdentity/model_2/embedding_2/embedding_lookup_1:output:0*
T0*=
_class3
1/loc:@model_2/embedding_2/embedding_lookup/56327*+
_output_shapes
:??????????
1model_2/embedding_2/embedding_lookup_1/Identity_1Identity8model_2/embedding_2/embedding_lookup_1/Identity:output:0*
T0*+
_output_shapes
:?????????c
!model_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model_2/concatenate_2/concatConcatV28model_2/embedding_2/embedding_lookup/Identity_1:output:0:model_2/embedding_2/embedding_lookup_1/Identity_1:output:0*model_2/concatenate_2/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????{
9model_2/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
'model_2/global_average_pooling1d_2/MeanMean%model_2/concatenate_2/concat:output:0Bmodel_2/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
%model_2/dense_5/MatMul/ReadVariableOpReadVariableOp.model_2_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_2/dense_5/MatMulMatMul0model_2/global_average_pooling1d_2/Mean:output:0-model_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_2/dense_5/BiasAddBiasAdd model_2/dense_5/MatMul:product:0.model_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
model_2/dense_5/ReluRelu model_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????|
model_2/dropout_2/IdentityIdentity"model_2/dense_5/Relu:activations:0*
T0*'
_output_shapes
:??????????
%model_2/dense_4/MatMul/ReadVariableOpReadVariableOp.model_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_2/dense_4/MatMulMatMul#model_2/dropout_2/Identity:output:0-model_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_2/dense_4/BiasAddBiasAdd model_2/dense_4/MatMul:product:0.model_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
model_2/dense_4/SigmoidSigmoid model_2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????j
IdentityIdentitymodel_2/dense_4/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp'^model_2/dense_4/BiasAdd/ReadVariableOp&^model_2/dense_4/MatMul/ReadVariableOp'^model_2/dense_5/BiasAdd/ReadVariableOp&^model_2/dense_5/MatMul/ReadVariableOp%^model_2/embedding_2/embedding_lookup'^model_2/embedding_2/embedding_lookup_1K^model_2/text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2M^model_2/text_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????:?????????: : : : : : : : : 2P
&model_2/dense_4/BiasAdd/ReadVariableOp&model_2/dense_4/BiasAdd/ReadVariableOp2N
%model_2/dense_4/MatMul/ReadVariableOp%model_2/dense_4/MatMul/ReadVariableOp2P
&model_2/dense_5/BiasAdd/ReadVariableOp&model_2/dense_5/BiasAdd/ReadVariableOp2N
%model_2/dense_5/MatMul/ReadVariableOp%model_2/dense_5/MatMul/ReadVariableOp2L
$model_2/embedding_2/embedding_lookup$model_2/embedding_2/embedding_lookup2P
&model_2/embedding_2/embedding_lookup_1&model_2/embedding_2/embedding_lookup_12?
Jmodel_2/text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2Jmodel_2/text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV22?
Lmodel_2/text_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2Lmodel_2/text_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_57547

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_56366

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_56594

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
V
:__inference_global_average_pooling1d_2_layer_call_fn_57463

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_56366i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_56512

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_restore_fn_57607
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
?
__inference_<lambda>_576228
4key_value_init45973_lookuptableimportv2_table_handle0
,key_value_init45973_lookuptableimportv2_keys2
.key_value_init45973_lookuptableimportv2_values	
identity??'key_value_init45973/LookupTableImportV2?
'key_value_init45973/LookupTableImportV2LookupTableImportV24key_value_init45973_lookuptableimportv2_table_handle,key_value_init45973_lookuptableimportv2_keys.key_value_init45973_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init45973/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2R
'key_value_init45973/LookupTableImportV2'key_value_init45973/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
+__inference_embedding_2_layer_call_fn_57436

inputs	
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_56479s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
*
__inference_<lambda>_57627
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_57381
inputs_0
inputs_1S
Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_2_string_lookup_2_equal_y3
/text_vectorization_2_string_lookup_2_selectv2_t	5
"embedding_2_embedding_lookup_57345:	?8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?embedding_2/embedding_lookup?embedding_2/embedding_lookup_1?Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2?Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2b
 text_vectorization_2/StringLowerStringLowerinputs_1*'
_output_shapes
:??????????
'text_vectorization_2/StaticRegexReplaceStaticRegexReplace)text_vectorization_2/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_2/SqueezeSqueeze0text_vectorization_2/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_2/StringSplit/StringSplitV2StringSplitV2%text_vectorization_2/Squeeze:output:0/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_2/StringSplit/strided_sliceStridedSlice8text_vectorization_2/StringSplit/StringSplitV2:indices:0=text_vectorization_2/StringSplit/strided_slice/stack:output:0?text_vectorization_2/StringSplit/strided_slice/stack_1:output:0?text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_2/StringSplit/strided_slice_1StridedSlice6text_vectorization_2/StringSplit/StringSplitV2:shape:0?text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle7text_vectorization_2/StringSplit/StringSplitV2:values:0Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_2/string_lookup_2/EqualEqual7text_vectorization_2/StringSplit/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/SelectV2SelectV2.text_vectorization_2/string_lookup_2/Equal:z:0/text_vectorization_2_string_lookup_2_selectv2_tKtext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/IdentityIdentity6text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
8text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_2/RaggedToTensor/Const:output:06text_vectorization_2/string_lookup_2/Identity:output:0:text_vectorization_2/RaggedToTensor/default_value:output:09text_vectorization_2/StringSplit/strided_slice_1:output:07text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSd
"text_vectorization_2/StringLower_1StringLowerinputs_0*'
_output_shapes
:??????????
)text_vectorization_2/StaticRegexReplace_1StaticRegexReplace+text_vectorization_2/StringLower_1:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_2/Squeeze_1Squeeze2text_vectorization_2/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????i
(text_vectorization_2/StringSplit_1/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
0text_vectorization_2/StringSplit_1/StringSplitV2StringSplitV2'text_vectorization_2/Squeeze_1:output:01text_vectorization_2/StringSplit_1/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
6text_vectorization_2/StringSplit_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
8text_vectorization_2/StringSplit_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
8text_vectorization_2/StringSplit_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
0text_vectorization_2/StringSplit_1/strided_sliceStridedSlice:text_vectorization_2/StringSplit_1/StringSplitV2:indices:0?text_vectorization_2/StringSplit_1/strided_slice/stack:output:0Atext_vectorization_2/StringSplit_1/strided_slice/stack_1:output:0Atext_vectorization_2/StringSplit_1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
8text_vectorization_2/StringSplit_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:text_vectorization_2/StringSplit_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:text_vectorization_2/StringSplit_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2text_vectorization_2/StringSplit_1/strided_slice_1StridedSlice8text_vectorization_2/StringSplit_1/StringSplitV2:shape:0Atext_vectorization_2/StringSplit_1/strided_slice_1/stack:output:0Ctext_vectorization_2/StringSplit_1/strided_slice_1/stack_1:output:0Ctext_vectorization_2/StringSplit_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Ytext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast9text_vectorization_2/StringSplit_1/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast;text_vectorization_2/StringSplit_1/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
btext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
gtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterktext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ptext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
btext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastitext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ntext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2jtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulftext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum_text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum_text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
ftext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ntext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsummtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
dtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
`text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2mtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2LookupTableFindV2Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle9text_vectorization_2/StringSplit_1/StringSplitV2:values:0Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_valueC^text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,text_vectorization_2/string_lookup_2/Equal_1Equal9text_vectorization_2/StringSplit_1/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
/text_vectorization_2/string_lookup_2/SelectV2_1SelectV20text_vectorization_2/string_lookup_2/Equal_1:z:0/text_vectorization_2_string_lookup_2_selectv2_tMtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/text_vectorization_2/string_lookup_2/Identity_1Identity8text_vectorization_2/string_lookup_2/SelectV2_1:output:0*
T0	*#
_output_shapes
:?????????u
3text_vectorization_2/RaggedToTensor_1/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
+text_vectorization_2/RaggedToTensor_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
:text_vectorization_2/RaggedToTensor_1/RaggedTensorToTensorRaggedTensorToTensor4text_vectorization_2/RaggedToTensor_1/Const:output:08text_vectorization_2/string_lookup_2/Identity_1:output:0<text_vectorization_2/RaggedToTensor_1/default_value:output:0;text_vectorization_2/StringSplit_1/strided_slice_1:output:09text_vectorization_2/StringSplit_1/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_57345Ctext_vectorization_2/RaggedToTensor_1/RaggedTensorToTensor:result:0*
Tindices0	*5
_class+
)'loc:@embedding_2/embedding_lookup/57345*+
_output_shapes
:?????????*
dtype0?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/57345*+
_output_shapes
:??????????
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
embedding_2/embedding_lookup_1ResourceGather"embedding_2_embedding_lookup_57345Atext_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*5
_class+
)'loc:@embedding_2/embedding_lookup/57345*+
_output_shapes
:?????????*
dtype0?
'embedding_2/embedding_lookup_1/IdentityIdentity'embedding_2/embedding_lookup_1:output:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/57345*+
_output_shapes
:??????????
)embedding_2/embedding_lookup_1/Identity_1Identity0embedding_2/embedding_lookup_1/Identity:output:0*
T0*+
_output_shapes
:?????????[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_2/concatConcatV20embedding_2/embedding_lookup/Identity_1:output:02embedding_2/embedding_lookup_1/Identity_1:output:0"concatenate_2/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????s
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_2/MeanMeanconcatenate_2/concat:output:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_5/MatMulMatMul(global_average_pooling1d_2/Mean:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_2/dropout/MulMuldense_5/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????a
dropout_2/dropout/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_4/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_4/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_2/embedding_lookup^embedding_2/embedding_lookup_1C^text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2E^text_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????:?????????: : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2@
embedding_2/embedding_lookup_1embedding_2/embedding_lookup_12?
Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV22?
Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
F
__inference__creator_57570
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_37314*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_57474

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_57480

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????U
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_56499

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????U
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_57527

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_57044
input_5
input_6S
Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_2_string_lookup_2_equal_y3
/text_vectorization_2_string_lookup_2_selectv2_t	$
embedding_2_57025:	?
dense_5_57032:
dense_5_57034:
dense_4_57038:
dense_4_57040:
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?%embedding_2/StatefulPartitionedCall_1?Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2?Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2a
 text_vectorization_2/StringLowerStringLowerinput_6*'
_output_shapes
:??????????
'text_vectorization_2/StaticRegexReplaceStaticRegexReplace)text_vectorization_2/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_2/SqueezeSqueeze0text_vectorization_2/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_2/StringSplit/StringSplitV2StringSplitV2%text_vectorization_2/Squeeze:output:0/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_2/StringSplit/strided_sliceStridedSlice8text_vectorization_2/StringSplit/StringSplitV2:indices:0=text_vectorization_2/StringSplit/strided_slice/stack:output:0?text_vectorization_2/StringSplit/strided_slice/stack_1:output:0?text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_2/StringSplit/strided_slice_1StridedSlice6text_vectorization_2/StringSplit/StringSplitV2:shape:0?text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle7text_vectorization_2/StringSplit/StringSplitV2:values:0Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_2/string_lookup_2/EqualEqual7text_vectorization_2/StringSplit/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/SelectV2SelectV2.text_vectorization_2/string_lookup_2/Equal:z:0/text_vectorization_2_string_lookup_2_selectv2_tKtext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/IdentityIdentity6text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
8text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_2/RaggedToTensor/Const:output:06text_vectorization_2/string_lookup_2/Identity:output:0:text_vectorization_2/RaggedToTensor/default_value:output:09text_vectorization_2/StringSplit/strided_slice_1:output:07text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSc
"text_vectorization_2/StringLower_1StringLowerinput_5*'
_output_shapes
:??????????
)text_vectorization_2/StaticRegexReplace_1StaticRegexReplace+text_vectorization_2/StringLower_1:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_2/Squeeze_1Squeeze2text_vectorization_2/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????i
(text_vectorization_2/StringSplit_1/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
0text_vectorization_2/StringSplit_1/StringSplitV2StringSplitV2'text_vectorization_2/Squeeze_1:output:01text_vectorization_2/StringSplit_1/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
6text_vectorization_2/StringSplit_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
8text_vectorization_2/StringSplit_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
8text_vectorization_2/StringSplit_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
0text_vectorization_2/StringSplit_1/strided_sliceStridedSlice:text_vectorization_2/StringSplit_1/StringSplitV2:indices:0?text_vectorization_2/StringSplit_1/strided_slice/stack:output:0Atext_vectorization_2/StringSplit_1/strided_slice/stack_1:output:0Atext_vectorization_2/StringSplit_1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
8text_vectorization_2/StringSplit_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:text_vectorization_2/StringSplit_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:text_vectorization_2/StringSplit_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2text_vectorization_2/StringSplit_1/strided_slice_1StridedSlice8text_vectorization_2/StringSplit_1/StringSplitV2:shape:0Atext_vectorization_2/StringSplit_1/strided_slice_1/stack:output:0Ctext_vectorization_2/StringSplit_1/strided_slice_1/stack_1:output:0Ctext_vectorization_2/StringSplit_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Ytext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast9text_vectorization_2/StringSplit_1/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast;text_vectorization_2/StringSplit_1/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
btext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
gtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterktext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ptext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
btext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastitext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ntext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2jtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulftext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum_text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum_text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
ftext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ntext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsummtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
dtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
`text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2mtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2LookupTableFindV2Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle9text_vectorization_2/StringSplit_1/StringSplitV2:values:0Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_valueC^text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,text_vectorization_2/string_lookup_2/Equal_1Equal9text_vectorization_2/StringSplit_1/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
/text_vectorization_2/string_lookup_2/SelectV2_1SelectV20text_vectorization_2/string_lookup_2/Equal_1:z:0/text_vectorization_2_string_lookup_2_selectv2_tMtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/text_vectorization_2/string_lookup_2/Identity_1Identity8text_vectorization_2/string_lookup_2/SelectV2_1:output:0*
T0	*#
_output_shapes
:?????????u
3text_vectorization_2/RaggedToTensor_1/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
+text_vectorization_2/RaggedToTensor_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
:text_vectorization_2/RaggedToTensor_1/RaggedTensorToTensorRaggedTensorToTensor4text_vectorization_2/RaggedToTensor_1/Const:output:08text_vectorization_2/string_lookup_2/Identity_1:output:0<text_vectorization_2/RaggedToTensor_1/default_value:output:0;text_vectorization_2/StringSplit_1/strided_slice_1:output:09text_vectorization_2/StringSplit_1/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallCtext_vectorization_2/RaggedToTensor_1/RaggedTensorToTensor:result:0embedding_2_57025*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_56479?
%embedding_2/StatefulPartitionedCall_1StatefulPartitionedCallAtext_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0embedding_2_57025*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_56479?
concatenate_2/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0.embedding_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_56492?
*global_average_pooling1d_2/PartitionedCallPartitionedCall&concatenate_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_56499?
dense_5/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0dense_5_57032dense_5_57034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_56512?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_56594?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_57038dense_4_57040*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_56536w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall&^embedding_2/StatefulPartitionedCall_1C^text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2E^text_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????:?????????: : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2N
%embedding_2/StatefulPartitionedCall_1%embedding_2/StatefulPartitionedCall_12?
Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV22?
Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_dense_5_layer_call_fn_57489

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_56512o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_2_layer_call_fn_57505

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_56523`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_embedding_2_layer_call_and_return_conditional_losses_56479

inputs	)
embedding_lookup_56473:	?
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_56473inputs*
Tindices0	*)
_class
loc:@embedding_lookup/56473*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/56473*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
r
H__inference_concatenate_2_layer_call_and_return_conditional_losses_56492

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :y
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_56543

inputs
inputs_1S
Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_2_string_lookup_2_equal_y3
/text_vectorization_2_string_lookup_2_selectv2_t	$
embedding_2_56480:	?
dense_5_56513:
dense_5_56515:
dense_4_56537:
dense_4_56539:
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?%embedding_2/StatefulPartitionedCall_1?Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2?Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2b
 text_vectorization_2/StringLowerStringLowerinputs_1*'
_output_shapes
:??????????
'text_vectorization_2/StaticRegexReplaceStaticRegexReplace)text_vectorization_2/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_2/SqueezeSqueeze0text_vectorization_2/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_2/StringSplit/StringSplitV2StringSplitV2%text_vectorization_2/Squeeze:output:0/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_2/StringSplit/strided_sliceStridedSlice8text_vectorization_2/StringSplit/StringSplitV2:indices:0=text_vectorization_2/StringSplit/strided_slice/stack:output:0?text_vectorization_2/StringSplit/strided_slice/stack_1:output:0?text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_2/StringSplit/strided_slice_1StridedSlice6text_vectorization_2/StringSplit/StringSplitV2:shape:0?text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle7text_vectorization_2/StringSplit/StringSplitV2:values:0Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_2/string_lookup_2/EqualEqual7text_vectorization_2/StringSplit/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/SelectV2SelectV2.text_vectorization_2/string_lookup_2/Equal:z:0/text_vectorization_2_string_lookup_2_selectv2_tKtext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/IdentityIdentity6text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
8text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_2/RaggedToTensor/Const:output:06text_vectorization_2/string_lookup_2/Identity:output:0:text_vectorization_2/RaggedToTensor/default_value:output:09text_vectorization_2/StringSplit/strided_slice_1:output:07text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSb
"text_vectorization_2/StringLower_1StringLowerinputs*'
_output_shapes
:??????????
)text_vectorization_2/StaticRegexReplace_1StaticRegexReplace+text_vectorization_2/StringLower_1:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_2/Squeeze_1Squeeze2text_vectorization_2/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????i
(text_vectorization_2/StringSplit_1/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
0text_vectorization_2/StringSplit_1/StringSplitV2StringSplitV2'text_vectorization_2/Squeeze_1:output:01text_vectorization_2/StringSplit_1/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
6text_vectorization_2/StringSplit_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
8text_vectorization_2/StringSplit_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
8text_vectorization_2/StringSplit_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
0text_vectorization_2/StringSplit_1/strided_sliceStridedSlice:text_vectorization_2/StringSplit_1/StringSplitV2:indices:0?text_vectorization_2/StringSplit_1/strided_slice/stack:output:0Atext_vectorization_2/StringSplit_1/strided_slice/stack_1:output:0Atext_vectorization_2/StringSplit_1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
8text_vectorization_2/StringSplit_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:text_vectorization_2/StringSplit_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:text_vectorization_2/StringSplit_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2text_vectorization_2/StringSplit_1/strided_slice_1StridedSlice8text_vectorization_2/StringSplit_1/StringSplitV2:shape:0Atext_vectorization_2/StringSplit_1/strided_slice_1/stack:output:0Ctext_vectorization_2/StringSplit_1/strided_slice_1/stack_1:output:0Ctext_vectorization_2/StringSplit_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Ytext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast9text_vectorization_2/StringSplit_1/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast;text_vectorization_2/StringSplit_1/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
btext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
gtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterktext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ptext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
btext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastitext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ntext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2jtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulftext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum_text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum_text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
ftext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ntext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsummtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
dtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
`text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2mtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2LookupTableFindV2Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle9text_vectorization_2/StringSplit_1/StringSplitV2:values:0Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_valueC^text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,text_vectorization_2/string_lookup_2/Equal_1Equal9text_vectorization_2/StringSplit_1/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
/text_vectorization_2/string_lookup_2/SelectV2_1SelectV20text_vectorization_2/string_lookup_2/Equal_1:z:0/text_vectorization_2_string_lookup_2_selectv2_tMtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/text_vectorization_2/string_lookup_2/Identity_1Identity8text_vectorization_2/string_lookup_2/SelectV2_1:output:0*
T0	*#
_output_shapes
:?????????u
3text_vectorization_2/RaggedToTensor_1/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
+text_vectorization_2/RaggedToTensor_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
:text_vectorization_2/RaggedToTensor_1/RaggedTensorToTensorRaggedTensorToTensor4text_vectorization_2/RaggedToTensor_1/Const:output:08text_vectorization_2/string_lookup_2/Identity_1:output:0<text_vectorization_2/RaggedToTensor_1/default_value:output:0;text_vectorization_2/StringSplit_1/strided_slice_1:output:09text_vectorization_2/StringSplit_1/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallCtext_vectorization_2/RaggedToTensor_1/RaggedTensorToTensor:result:0embedding_2_56480*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_56479?
%embedding_2/StatefulPartitionedCall_1StatefulPartitionedCallAtext_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0embedding_2_56480*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_56479?
concatenate_2/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0.embedding_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_56492?
*global_average_pooling1d_2/PartitionedCallPartitionedCall&concatenate_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_56499?
dense_5/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0dense_5_56513dense_5_56515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_56512?
dropout_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_56523?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_4_56537dense_4_56539*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_56536w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall&^embedding_2/StatefulPartitionedCall_1C^text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2E^text_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????:?????????: : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2N
%embedding_2/StatefulPartitionedCall_1%embedding_2/StatefulPartitionedCall_12?
Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV22?
Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_model_2_layer_call_fn_57124
inputs_0
inputs_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_56769o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_model_2_layer_call_fn_56564
input_5
input_6
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_56543o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_dense_4_layer_call_fn_57536

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_56536o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
,
__inference__destroyer_57565
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
#__inference_signature_wrapper_57076
input_5
input_6
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_56356o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_56769

inputs
inputs_1S
Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_2_string_lookup_2_equal_y3
/text_vectorization_2_string_lookup_2_selectv2_t	$
embedding_2_56750:	?
dense_5_56757:
dense_5_56759:
dense_4_56763:
dense_4_56765:
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?%embedding_2/StatefulPartitionedCall_1?Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2?Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2b
 text_vectorization_2/StringLowerStringLowerinputs_1*'
_output_shapes
:??????????
'text_vectorization_2/StaticRegexReplaceStaticRegexReplace)text_vectorization_2/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_2/SqueezeSqueeze0text_vectorization_2/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_2/StringSplit/StringSplitV2StringSplitV2%text_vectorization_2/Squeeze:output:0/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_2/StringSplit/strided_sliceStridedSlice8text_vectorization_2/StringSplit/StringSplitV2:indices:0=text_vectorization_2/StringSplit/strided_slice/stack:output:0?text_vectorization_2/StringSplit/strided_slice/stack_1:output:0?text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_2/StringSplit/strided_slice_1StridedSlice6text_vectorization_2/StringSplit/StringSplitV2:shape:0?text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle7text_vectorization_2/StringSplit/StringSplitV2:values:0Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_2/string_lookup_2/EqualEqual7text_vectorization_2/StringSplit/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/SelectV2SelectV2.text_vectorization_2/string_lookup_2/Equal:z:0/text_vectorization_2_string_lookup_2_selectv2_tKtext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/IdentityIdentity6text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
8text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_2/RaggedToTensor/Const:output:06text_vectorization_2/string_lookup_2/Identity:output:0:text_vectorization_2/RaggedToTensor/default_value:output:09text_vectorization_2/StringSplit/strided_slice_1:output:07text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSb
"text_vectorization_2/StringLower_1StringLowerinputs*'
_output_shapes
:??????????
)text_vectorization_2/StaticRegexReplace_1StaticRegexReplace+text_vectorization_2/StringLower_1:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_2/Squeeze_1Squeeze2text_vectorization_2/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????i
(text_vectorization_2/StringSplit_1/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
0text_vectorization_2/StringSplit_1/StringSplitV2StringSplitV2'text_vectorization_2/Squeeze_1:output:01text_vectorization_2/StringSplit_1/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
6text_vectorization_2/StringSplit_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
8text_vectorization_2/StringSplit_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
8text_vectorization_2/StringSplit_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
0text_vectorization_2/StringSplit_1/strided_sliceStridedSlice:text_vectorization_2/StringSplit_1/StringSplitV2:indices:0?text_vectorization_2/StringSplit_1/strided_slice/stack:output:0Atext_vectorization_2/StringSplit_1/strided_slice/stack_1:output:0Atext_vectorization_2/StringSplit_1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
8text_vectorization_2/StringSplit_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:text_vectorization_2/StringSplit_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:text_vectorization_2/StringSplit_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2text_vectorization_2/StringSplit_1/strided_slice_1StridedSlice8text_vectorization_2/StringSplit_1/StringSplitV2:shape:0Atext_vectorization_2/StringSplit_1/strided_slice_1/stack:output:0Ctext_vectorization_2/StringSplit_1/strided_slice_1/stack_1:output:0Ctext_vectorization_2/StringSplit_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Ytext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast9text_vectorization_2/StringSplit_1/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast;text_vectorization_2/StringSplit_1/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
btext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
gtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterktext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ptext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
btext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastitext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ntext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2jtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulftext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum_text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum_text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
ftext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ntext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsummtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
dtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
`text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2mtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2LookupTableFindV2Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle9text_vectorization_2/StringSplit_1/StringSplitV2:values:0Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_valueC^text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,text_vectorization_2/string_lookup_2/Equal_1Equal9text_vectorization_2/StringSplit_1/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
/text_vectorization_2/string_lookup_2/SelectV2_1SelectV20text_vectorization_2/string_lookup_2/Equal_1:z:0/text_vectorization_2_string_lookup_2_selectv2_tMtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/text_vectorization_2/string_lookup_2/Identity_1Identity8text_vectorization_2/string_lookup_2/SelectV2_1:output:0*
T0	*#
_output_shapes
:?????????u
3text_vectorization_2/RaggedToTensor_1/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
+text_vectorization_2/RaggedToTensor_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
:text_vectorization_2/RaggedToTensor_1/RaggedTensorToTensorRaggedTensorToTensor4text_vectorization_2/RaggedToTensor_1/Const:output:08text_vectorization_2/string_lookup_2/Identity_1:output:0<text_vectorization_2/RaggedToTensor_1/default_value:output:0;text_vectorization_2/StringSplit_1/strided_slice_1:output:09text_vectorization_2/StringSplit_1/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallCtext_vectorization_2/RaggedToTensor_1/RaggedTensorToTensor:result:0embedding_2_56750*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_56479?
%embedding_2/StatefulPartitionedCall_1StatefulPartitionedCallAtext_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0embedding_2_56750*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_56479?
concatenate_2/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0.embedding_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_56492?
*global_average_pooling1d_2/PartitionedCallPartitionedCall&concatenate_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_56499?
dense_5/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0dense_5_56757dense_5_56759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_56512?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_56594?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_56763dense_4_56765*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_56536w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall&^embedding_2/StatefulPartitionedCall_1C^text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2E^text_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????:?????????: : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2N
%embedding_2/StatefulPartitionedCall_1%embedding_2/StatefulPartitionedCall_12?
Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV22?
Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_57500

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
:
__inference__creator_57552
identity??
hash_tablem

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name45974*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?;
?
__inference__traced_save_57737
file_prefix5
1savev2_embedding_2_embeddings_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop<
8savev2_adam_embedding_2_embeddings_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop<
8savev2_adam_embedding_2_embeddings_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_2_embeddings_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop8savev2_adam_embedding_2_embeddings_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop8savev2_adam_embedding_2_embeddings_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *)
dtypes
2		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?::::: : : : : ::: : : : :	?:::::	?::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	?:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_57249
inputs_0
inputs_1S
Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_2_string_lookup_2_equal_y3
/text_vectorization_2_string_lookup_2_selectv2_t	5
"embedding_2_embedding_lookup_57220:	?8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?embedding_2/embedding_lookup?embedding_2/embedding_lookup_1?Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2?Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2b
 text_vectorization_2/StringLowerStringLowerinputs_1*'
_output_shapes
:??????????
'text_vectorization_2/StaticRegexReplaceStaticRegexReplace)text_vectorization_2/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_2/SqueezeSqueeze0text_vectorization_2/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_2/StringSplit/StringSplitV2StringSplitV2%text_vectorization_2/Squeeze:output:0/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_2/StringSplit/strided_sliceStridedSlice8text_vectorization_2/StringSplit/StringSplitV2:indices:0=text_vectorization_2/StringSplit/strided_slice/stack:output:0?text_vectorization_2/StringSplit/strided_slice/stack_1:output:0?text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_2/StringSplit/strided_slice_1StridedSlice6text_vectorization_2/StringSplit/StringSplitV2:shape:0?text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle7text_vectorization_2/StringSplit/StringSplitV2:values:0Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_2/string_lookup_2/EqualEqual7text_vectorization_2/StringSplit/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/SelectV2SelectV2.text_vectorization_2/string_lookup_2/Equal:z:0/text_vectorization_2_string_lookup_2_selectv2_tKtext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/IdentityIdentity6text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
8text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_2/RaggedToTensor/Const:output:06text_vectorization_2/string_lookup_2/Identity:output:0:text_vectorization_2/RaggedToTensor/default_value:output:09text_vectorization_2/StringSplit/strided_slice_1:output:07text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSd
"text_vectorization_2/StringLower_1StringLowerinputs_0*'
_output_shapes
:??????????
)text_vectorization_2/StaticRegexReplace_1StaticRegexReplace+text_vectorization_2/StringLower_1:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_2/Squeeze_1Squeeze2text_vectorization_2/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????i
(text_vectorization_2/StringSplit_1/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
0text_vectorization_2/StringSplit_1/StringSplitV2StringSplitV2'text_vectorization_2/Squeeze_1:output:01text_vectorization_2/StringSplit_1/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
6text_vectorization_2/StringSplit_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
8text_vectorization_2/StringSplit_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
8text_vectorization_2/StringSplit_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
0text_vectorization_2/StringSplit_1/strided_sliceStridedSlice:text_vectorization_2/StringSplit_1/StringSplitV2:indices:0?text_vectorization_2/StringSplit_1/strided_slice/stack:output:0Atext_vectorization_2/StringSplit_1/strided_slice/stack_1:output:0Atext_vectorization_2/StringSplit_1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
8text_vectorization_2/StringSplit_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:text_vectorization_2/StringSplit_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:text_vectorization_2/StringSplit_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2text_vectorization_2/StringSplit_1/strided_slice_1StridedSlice8text_vectorization_2/StringSplit_1/StringSplitV2:shape:0Atext_vectorization_2/StringSplit_1/strided_slice_1/stack:output:0Ctext_vectorization_2/StringSplit_1/strided_slice_1/stack_1:output:0Ctext_vectorization_2/StringSplit_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Ytext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast9text_vectorization_2/StringSplit_1/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast;text_vectorization_2/StringSplit_1/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
btext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
gtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterktext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ptext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
btext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastitext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ntext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2jtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ltext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulftext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum_text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum_text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
ftext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount]text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ntext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsummtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
dtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
`text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[text_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2mtext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0atext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0itext_vectorization_2/StringSplit_1/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2LookupTableFindV2Otext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle9text_vectorization_2/StringSplit_1/StringSplitV2:values:0Ptext_vectorization_2_string_lookup_2_none_lookup_lookuptablefindv2_default_valueC^text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,text_vectorization_2/string_lookup_2/Equal_1Equal9text_vectorization_2/StringSplit_1/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
/text_vectorization_2/string_lookup_2/SelectV2_1SelectV20text_vectorization_2/string_lookup_2/Equal_1:z:0/text_vectorization_2_string_lookup_2_selectv2_tMtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/text_vectorization_2/string_lookup_2/Identity_1Identity8text_vectorization_2/string_lookup_2/SelectV2_1:output:0*
T0	*#
_output_shapes
:?????????u
3text_vectorization_2/RaggedToTensor_1/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
+text_vectorization_2/RaggedToTensor_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
:text_vectorization_2/RaggedToTensor_1/RaggedTensorToTensorRaggedTensorToTensor4text_vectorization_2/RaggedToTensor_1/Const:output:08text_vectorization_2/string_lookup_2/Identity_1:output:0<text_vectorization_2/RaggedToTensor_1/default_value:output:0;text_vectorization_2/StringSplit_1/strided_slice_1:output:09text_vectorization_2/StringSplit_1/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_57220Ctext_vectorization_2/RaggedToTensor_1/RaggedTensorToTensor:result:0*
Tindices0	*5
_class+
)'loc:@embedding_2/embedding_lookup/57220*+
_output_shapes
:?????????*
dtype0?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/57220*+
_output_shapes
:??????????
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
embedding_2/embedding_lookup_1ResourceGather"embedding_2_embedding_lookup_57220Atext_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*5
_class+
)'loc:@embedding_2/embedding_lookup/57220*+
_output_shapes
:?????????*
dtype0?
'embedding_2/embedding_lookup_1/IdentityIdentity'embedding_2/embedding_lookup_1:output:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/57220*+
_output_shapes
:??????????
)embedding_2/embedding_lookup_1/Identity_1Identity0embedding_2/embedding_lookup_1/Identity:output:0*
T0*+
_output_shapes
:?????????[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_2/concatConcatV20embedding_2/embedding_lookup/Identity_1:output:02embedding_2/embedding_lookup_1/Identity_1:output:0"concatenate_2/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????s
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_2/MeanMeanconcatenate_2/concat:output:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_5/MatMulMatMul(global_average_pooling1d_2/Mean:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????l
dropout_2/IdentityIdentitydense_5/Relu:activations:0*
T0*'
_output_shapes
:??????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_4/MatMulMatMuldropout_2/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_4/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_2/embedding_lookup^embedding_2/embedding_lookup_1C^text_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2E^text_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????:?????????: : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2@
embedding_2/embedding_lookup_1embedding_2/embedding_lookup_12?
Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV2Btext_vectorization_2/string_lookup_2/None_Lookup/LookupTableFindV22?
Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2Dtext_vectorization_2/string_lookup_2/None_Lookup_1/LookupTableFindV2:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_56536

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_56523

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_2_layer_call_fn_57451
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_56492d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
t
H__inference_concatenate_2_layer_call_and_return_conditional_losses_57458
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :{
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
b
)__inference_dropout_2_layer_call_fn_57510

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_56594o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_50
serving_default_input_5:0?????????
;
input_60
serving_default_input_6:0?????????=
dense_42
StatefulPartitionedCall_1:0?????????tensorflow/serving/predict:җ
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
p__call__
*q&call_and_return_all_conditional_losses
r_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
P
_lookup_layer
	keras_api
s_adapt_function"
_tf_keras_layer
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
?

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
/iter

0beta_1

1beta_2
	2decay
3learning_ratemfmg mh)mi*mjvkvl vm)vn*vo"
	optimizer
C
1
2
 3
)4
*5"
trackable_list_wrapper
C
0
1
 2
)3
*4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
p__call__
r_default_save_signature
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
L
9lookup_table
:token_counts
;	keras_api"
_tf_keras_layer
"
_generic_user_object
):'	?2embedding_2/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 :2dense_5/kernel
:2dense_5/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
!	variables
"trainable_variables
#regularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
%	variables
&trainable_variables
'regularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 :2dense_4/kernel
:2dense_4/bias
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
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
+	variables
,trainable_variables
-regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
m
\_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
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
N
	]total
	^count
_	variables
`	keras_api"
_tf_keras_metric
^
	atotal
	bcount
c
_fn_kwargs
d	variables
e	keras_api"
_tf_keras_metric
"
_generic_user_object
:  (2total
:  (2count
.
]0
^1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
a0
b1"
trackable_list_wrapper
-
d	variables"
_generic_user_object
.:,	?2Adam/embedding_2/embeddings/m
%:#2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
%:#2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
.:,	?2Adam/embedding_2/embeddings/v
%:#2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
%:#2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
?2?
'__inference_model_2_layer_call_fn_56564
'__inference_model_2_layer_call_fn_57100
'__inference_model_2_layer_call_fn_57124
'__inference_model_2_layer_call_fn_56814?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_2_layer_call_and_return_conditional_losses_57249
B__inference_model_2_layer_call_and_return_conditional_losses_57381
B__inference_model_2_layer_call_and_return_conditional_losses_56929
B__inference_model_2_layer_call_and_return_conditional_losses_57044?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_56356input_5input_6"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_57429?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_embedding_2_layer_call_fn_57436?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_embedding_2_layer_call_and_return_conditional_losses_57445?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_2_layer_call_fn_57451?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_concatenate_2_layer_call_and_return_conditional_losses_57458?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
:__inference_global_average_pooling1d_2_layer_call_fn_57463
:__inference_global_average_pooling1d_2_layer_call_fn_57468?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_57474
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_57480?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_5_layer_call_fn_57489?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_5_layer_call_and_return_conditional_losses_57500?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dropout_2_layer_call_fn_57505
)__inference_dropout_2_layer_call_fn_57510?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_2_layer_call_and_return_conditional_losses_57515
D__inference_dropout_2_layer_call_and_return_conditional_losses_57527?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_4_layer_call_fn_57536?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_4_layer_call_and_return_conditional_losses_57547?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_57076input_5input_6"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_57552?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_57560?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_57565?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_57570?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_57575?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_57580?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_57599checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_57607restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_56
__inference__creator_57552?

? 
? "? 6
__inference__creator_57570?

? 
? "? 8
__inference__destroyer_57565?

? 
? "? 8
__inference__destroyer_57580?

? 
? "? A
__inference__initializer_575609???

? 
? "? :
__inference__initializer_57575?

? 
? "? ?
 __inference__wrapped_model_56356?9??? )*X?U
N?K
I?F
!?
input_5?????????
!?
input_6?????????
? "1?.
,
dense_4!?
dense_4?????????j
__inference_adapt_step_57429J:???<
5?2
0?-?
??????????IteratorSpec 
? "
 ?
H__inference_concatenate_2_layer_call_and_return_conditional_losses_57458?b?_
X?U
S?P
&?#
inputs/0?????????
&?#
inputs/1?????????
? ")?&
?
0?????????
? ?
-__inference_concatenate_2_layer_call_fn_57451?b?_
X?U
S?P
&?#
inputs/0?????????
&?#
inputs/1?????????
? "???????????
B__inference_dense_4_layer_call_and_return_conditional_losses_57547\)*/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_4_layer_call_fn_57536O)*/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_5_layer_call_and_return_conditional_losses_57500\ /?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_5_layer_call_fn_57489O /?,
%?"
 ?
inputs?????????
? "???????????
D__inference_dropout_2_layer_call_and_return_conditional_losses_57515\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_57527\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? |
)__inference_dropout_2_layer_call_fn_57505O3?0
)?&
 ?
inputs?????????
p 
? "??????????|
)__inference_dropout_2_layer_call_fn_57510O3?0
)?&
 ?
inputs?????????
p
? "???????????
F__inference_embedding_2_layer_call_and_return_conditional_losses_57445_/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????
? ?
+__inference_embedding_2_layer_call_fn_57436R/?,
%?"
 ?
inputs?????????	
? "???????????
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_57474{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_57480`7?4
-?*
$?!
inputs?????????

 
? "%?"
?
0?????????
? ?
:__inference_global_average_pooling1d_2_layer_call_fn_57463nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
:__inference_global_average_pooling1d_2_layer_call_fn_57468S7?4
-?*
$?!
inputs?????????

 
? "???????????
B__inference_model_2_layer_call_and_return_conditional_losses_56929?9??? )*`?]
V?S
I?F
!?
input_5?????????
!?
input_6?????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_57044?9??? )*`?]
V?S
I?F
!?
input_5?????????
!?
input_6?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_57249?9??? )*b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_57381?9??? )*b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
'__inference_model_2_layer_call_fn_56564?9??? )*`?]
V?S
I?F
!?
input_5?????????
!?
input_6?????????
p 

 
? "???????????
'__inference_model_2_layer_call_fn_56814?9??? )*`?]
V?S
I?F
!?
input_5?????????
!?
input_6?????????
p

 
? "???????????
'__inference_model_2_layer_call_fn_57100?9??? )*b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
'__inference_model_2_layer_call_fn_57124?9??? )*b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "??????????y
__inference_restore_fn_57607Y:K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_57599?:&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
#__inference_signature_wrapper_57076?9??? )*i?f
? 
_?\
,
input_5!?
input_5?????????
,
input_6!?
input_6?????????"1?.
,
dense_4!?
dense_4?????????