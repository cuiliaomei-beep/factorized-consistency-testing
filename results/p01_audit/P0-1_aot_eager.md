# P0-1 recomputation: frozen 185-program corpus x graph break after every statement (backend=aot_eager, torch 2.14.0+cpu, 2026-09-25 17:06)

One process, one oracle (exact comparison of return values + STATE snapshot + exception type), three arms: eager, compiled original program, compiled break variant. Every program is called twice.

## Counts (numerator and denominator reported separately)

| Quantity | Value |
|---|---|
| Frozen programs | 185 (dynamo_semantics 77, dynamo_semantics_more 74, dynamo_semantics_batch3 34) |
| Runnable / rewritable programs | 185 / 185 (0 AST rewrite failures) |
| Calls per arm | 370 (185 programs x 2 calls); 1110 over the three arms |
| Static insertion positions (after top-level statements) | 477; 11 programs have no insertion position |
| User graph breaks actually hit (Dynamo counter `graph_break`, both calls; can exceed the static count when a frame is retraced and fall below it when a statement is not reached) | 543 |
| Programs in which every static break was hit | 155; 19 programs hit fewer breaks than inserted |
| Graph segments (Dynamo `frames.total`, both calls) | original arm 724, break arm 1279 |
| Divergence instances (program x call) | original arm 21, break arm 27, intersection 19 |
| Paired classification | same (no divergence) 341; same divergence 13; new 8; gone 2; changed 6 |
| Wall time of this run | 68.2 s (eager arm 0.07 s, original arm 21.64 s, break arm 45.35 s, compilation included) |

## Pairs that are not 'same' (program x call)

| Program | Call | Pair | Original arm | Break arm | Static breaks | Breaks hit | Segments orig/break | Root cause | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `gen_send` | 0 | new | - | raise | 4 | 3 | 1 / 4 | C42 | defect |
| `gen_send` | 1 | new | - | raise | 4 | 3 | 1 / 4 | C42 | defect |
| `collections_types` | 0 | new | - | return | 9 | 9 | 1 / 10 | C41 | defect |
| `collections_types` | 1 | new | - | return | 9 | 9 | 1 / 10 | C41 | defect |
| `python_random` | 0 | gone | return | - | 4 | 4 | 3 / 6 | known-B17 | known-defect |
| `python_random` | 1 | gone | return | - | 4 | 4 | 3 / 6 | known-B17 | known-defect |
| `numpy_scalar_types` | 0 | changed | return | return | 2 | 2 | 2 / 3 | known-numpy | known-defect |
| `numpy_scalar_types` | 1 | changed | return | return | 2 | 2 | 2 / 3 | known-numpy | known-defect |
| `gen_return_value_stopiteration` | 0 | new | - | return | 4 | 4 | 1 / 5 | C42 | defect |
| `gen_return_value_stopiteration` | 1 | new | - | return | 4 | 4 | 1 / 5 | C42 | defect |
| `gen_throw` | 0 | new | - | raise | 3 | 3 | 1 / 4 | C42 | defect |
| `gen_throw` | 1 | new | - | raise | 3 | 3 | 1 / 4 | C42 | defect |
| `tensor_subclass_torch_function` | 0 | changed | state | raise | 3 | 3 | 8 / 6 | known-state | known-defect |
| `tensor_subclass_torch_function` | 1 | changed | state | raise | 3 | 3 | 8 / 6 | known-state | known-defect |
| `threading_local_state` | 0 | changed | state | state | 2 | 4 | 4 / 7 | known-state | known-defect |
| `threading_local_state` | 1 | changed | state | state | 2 | 4 | 4 / 7 | known-state | known-defect |

## Independent root causes (new divergences after deduplication)

8 new divergence instances in 4 programs, **2 independent root causes** after deduplication:

- **C42**: `gen_send`, `gen_return_value_stopiteration`, `gen_throw`: generator alive across the break is reconstructed as tuple_iterator; .send raises AttributeError (#198190)
- **C41**: `collections_types`: OrderedDict.move_to_end on a dict that becomes an input of the resume function is not replayed (#198189)

Gone and changed pairs are not new findings:
- `python_random` (gone): baseline diverges on both calls (values differ from eager) = the known item B17, random.seed inside a compiled function ignored (fixed upstream); with a break after every statement seed and draws are no longer in one traced frame and the compiled values equal eager, so the divergence disappears
- `numpy_scalar_types` (changed): aot_eager arm only: baseline already diverges (numpy scalar promotion, known family C13/C20); under the break the compiled dtype changes from float32 to float64, still a divergence
- `tensor_subclass_torch_function` (changed): baseline already diverges in the __torch_function__ call log; the break changes which calls are logged, not whether it diverges
- `threading_local_state` (changed): baseline already diverges on the threading.local object stored in STATE; the break changes the repr of the object, not the divergence

Expected behavior / invalid items: none among the new divergences of the 185 programs (the `functools.lru_cache` inlining belongs to batch-4 programs, which are not in the frozen corpus).

## Breaks and graph segments of the representative programs (Dynamo counters)

| Program | Static breaks | Breaks hit | Segments, original arm | Segments, break arm | Other break reasons (break arm) |
|---|---|---|---|---|---|
| `gen_send` | 4 | 3 | 1 | 4 | Unsupported method call |
| `collections_types` | 9 | 9 | 1 | 10 | - |
| `python_random` | 4 | 4 | 3 | 6 | Attempted to call function marked as skipped |
| `numpy_scalar_types` | 2 | 2 | 2 | 3 | - |
| `gen_return_value_stopiteration` | 4 | 4 | 1 | 5 | - |
| `gen_throw` | 3 | 3 | 1 | 4 | Unsupported method call |
| `tensor_subclass_torch_function` | 3 | 3 | 8 | 6 | Invalid call to __build_class__; Unsupported Tensor.item() call with capture_scalar_outputs=False |
| `threading_local_state` | 2 | 4 | 4 | 7 | Unsupported function call |

Programs that hit fewer breaks than inserted (the break follows a statement that is not reached, or the program raises before it):

- `gen_early_close`: inserted 4, hit 2
- `gen_send`: inserted 4, hit 3
- `assert_message`: inserted 1, hit 0
- `closure_late_binding`: inserted 2, hit 1
- `zip_strict_error`: inserted 1, hit 0
- `graph_break_in_loop`: inserted 3, hit 1
- `break_inside_try`: inserted 2, hit 0
- `list_of_tensors_equality`: inserted 1, hit 0
- `exception_message_with_value`: inserted 2, hit 1
- `exception_after_graph_break`: inserted 4, hit 3
- `exception_in_generator`: inserted 4, hit 2
- `exception_chaining`: inserted 1, hit 0
- `exception_group`: inserted 1, hit 0
- `while_with_tensor_condition_item`: inserted 3, hit 2
- `print_format_side_effect_count`: inserted 1, hit 0
- `gen_expression_lazy_side_effect`: inserted 4, hit 1
- `nested_generator_closure_var`: inserted 2, hit 1
- `exception_args_and_notes`: inserted 1, hit 0
- `nonlocal_generator_state_machine`: inserted 5, hit 1

## Do the minimal reproducers still need a graph break?

See `results/p01_audit/minimal/`. The C41 reproducer contains no `graph_break` and shows different OrderedDict orders in eager and compiled execution. The C42 reproducer has two parts: the first (a generator returned from the compiled function) reproduces `tuple_iterator` and `StopIteration.value=None` without any break; the second (`.send` on a suspended generator) uses one explicit `graph_break()` as the suspension point. The return path of C41 and C42 therefore needs no break; the in-frame send/throw form of C42 is triggered by a break.
