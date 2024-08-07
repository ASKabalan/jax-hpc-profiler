# Reporting for fcn
## Parameters
---------  -------
Function   fcn
Precision  float32
X          1000
Y          1000
Z          1000
PX         1
PY         1
Backend    NCCL
Nodes      1
done       yes
---------  -------
---
## Profiling Data
--------------  ----------------
JIT Time            87.0802
Min Time             0.331191
Max Time            34.4314
Mean Time            3.78202
Std Time            10.2165
Last Time            0.362222
Generated Code    3624
Argument Size        1.2e+07
Output Size          4e+06
Temporary Size       4.19443e+06
FLOPS           999999
--------------  ----------------
---
## Compiled Code
```hlo
HloModule jit_fcn, is_scheduled=true, entry_computation_layout={(f32[1000,1000]{1,0}, f32[1000,1000]{1,0}, f32[1000,1000]{1,0})->f32[1000,1000]{1,0}}, allow_spmd_sharding_propagation_to_parameters={true,true,true}, allow_spmd_sharding_propagation_to_output={true}, frontend_attributes={fingerprint_before_lhs="1cf366219d1ea540492ab9a2222c2268"}

%wrapped_add_computation (param_0: f32[1000,1000], param_1: f32[1000,1000]) -> f32[1000,1000] {
  %param_0 = f32[1000,1000]{1,0} parameter(0)
  %param_1 = f32[1000,1000]{1,0} parameter(1)
  ROOT %add.1.1 = f32[1000,1000]{1,0} add(f32[1000,1000]{1,0} %param_0, f32[1000,1000]{1,0} %param_1), metadata={op_name="jit(fcn)/jit(main)/add" source_file="/home/wassim/Projects/hpc-plotter/timer_sample/function.py" source_line=7}
}

ENTRY %main.6 (Arg_0.1.0: f32[1000,1000], Arg_1.2.0: f32[1000,1000], Arg_2.3.0: f32[1000,1000]) -> f32[1000,1000] {
  %Arg_2.3.0 = f32[1000,1000]{1,0} parameter(2), metadata={op_name="k"}
  %Arg_1.2.0 = f32[1000,1000]{1,0} parameter(1), metadata={op_name="n"}
  %Arg_0.1.0 = f32[1000,1000]{1,0} parameter(0), metadata={op_name="m"}
  %custom-call.1.0 = (f32[1000,1000]{1,0}, s8[4194304]{0}) custom-call(f32[1000,1000]{1,0} %Arg_0.1.0, f32[1000,1000]{1,0} %Arg_1.2.0), custom_call_target="__cublas$gemm", metadata={op_name="jit(fcn)/jit(main)/dot_general[dimension_numbers=(((1,), (0,)), ((), ())) precision=None preferred_element_type=float32]" source_file="/home/wassim/Projects/hpc-plotter/timer_sample/function.py" source_line=7}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","damax_output":false,"lhs_stride":"1000000","rhs_stride":"1000000","grad_x":false,"grad_y":false},"force_earliest_schedule":false}
  %get-tuple-element.1 = f32[1000,1000]{1,0} get-tuple-element((f32[1000,1000]{1,0}, s8[4194304]{0}) %custom-call.1.0), index=0, metadata={op_name="jit(fcn)/jit(main)/dot_general[dimension_numbers=(((1,), (0,)), ((), ())) precision=None preferred_element_type=float32]" source_file="/home/wassim/Projects/hpc-plotter/timer_sample/function.py" source_line=7}
  ROOT %wrapped_add = f32[1000,1000]{1,0} fusion(f32[1000,1000]{1,0} %get-tuple-element.1, f32[1000,1000]{1,0} %Arg_2.3.0), kind=kLoop, calls=%wrapped_add_computation, metadata={op_name="jit(fcn)/jit(main)/add" source_file="/home/wassim/Projects/hpc-plotter/timer_sample/function.py" source_line=7}
}


```

---
## Lowered Code
```hlo
module @jit_fcn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1000x1000xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<1000x1000xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<1000x1000xf32> {mhlo.layout_mode = "default"}) -> (tensor<1000x1000xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1000x1000xf32>, tensor<1000x1000xf32>) -> tensor<1000x1000xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<1000x1000xf32>
    return %1 : tensor<1000x1000xf32>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f32[1000,1000] b:f32[1000,1000] c:f32[1000,1000]. let
    d:f32[1000,1000] = dot_general[
      dimension_numbers=(([1], [0]), ([], []))
      preferred_element_type=float32
    ] a b
    e:f32[1000,1000] = add d c
  in (e,) }
```
