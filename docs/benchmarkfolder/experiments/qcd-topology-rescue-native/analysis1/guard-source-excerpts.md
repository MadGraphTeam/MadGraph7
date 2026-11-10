## historical_bridge

`.tirust-build/bridge-poles/src/lib.rs`

SHA256 `1eab09d61652e4dfade900a22f8c63a071b4fecee2e7454e8144f934f17e34e0`

```text
20: 
21: pub const ABI_VERSION: u32 = 1;
22: const SUCCESS: i32 = 0;
23: const UNSTABLE: i32 = 1;
24: const ERROR: i32 = 2;
25: const UNSUPPORTED: i32 = 3;
26: const RECORD: usize = 64;
27: const MP_BITS: u32 = 256;
28: const DEFAULT_BUDGET: u64 = 268_435_456;
29: type Key = (usize, u16);
```

```text
43:     budget: u64,
44:     plans: HashMap<Key, DirectReductionPlan>,
45:     dp: HashMap<Key, TensorWorkspace<f64>>,
46:     mp: HashMap<Key, DirectWorkspace<Float>>,
47:     store: TensorStore,
48:     qp_cache: QpCache,
49:     #[cfg(feature = "poles-only")]
50:     pole_dp: HashMap<Key, DirectPoleWorkspace<f64>>,
51:     #[cfg(feature = "poles-only")]
52:     pole_mp: HashMap<Key, DirectPoleWorkspace<Float>>,
```

```text
79:         if !self.plans.contains_key(&key) {
80:             let plan = DirectReductionPlan::new(settings(key)).map_err(|e| e.to_string())?;
81:             self.store.register(&plan);
82:             #[cfg(feature = "poles-only")]
83:             self.pole_store.register(&plan);
84:             self.plans.insert(key, plan);
85:         }
86:         Ok(())
87:     }
```

```text
152:     fn mp(
153:         &mut self,
154:         key: Key,
155:         batch: KinematicsBatch<Float>,
156:         mode: Mode,
157:     ) -> Result<(i32, Vec<u8>)> {
158:         self.register(key)?;
159:         if !mode.finite {
160:             #[cfg(feature = "poles-only")]
161:             return self.mp_poles(key, batch, mode);
162:             #[cfg(not(feature = "poles-only"))]
163:             return Ok((UNSUPPORTED, Vec::new()));
164:         }
165:         if !self.mp.contains_key(&key) {
166:             let mut workspace = self.plans[&key]
167:                 .workspace(Float::new(MP_BITS))
168:                 .map_err(|e| e.to_string())?;
169:             workspace.set_input_precision(113);
170:             // 256-bit arithmetic with substantial headroom above binary128.
171:             // The external MadLoop check still decides numerical acceptance.
172:             workspace.set_stability_threshold(Float::with_val(MP_BITS, 2).pow(136).inv());
173:             self.mp.insert(key, workspace);
174:         }
175:         // MPFR store reuse is not supported by TIRust; do not claim DP hits as QP reuse.
176:         let workspace = self.mp.get_mut(&key).unwrap();
177:         let result = workspace.reduce_batch(&batch).map_err(|e| e.to_string())?;
178:         match result.lane(0) {
179:             Ok(values) => Ok((SUCCESS, pack_mp(values, mode.poles, 4)?)),
180:             Err(status) => {
181:                 set_error(&format!("TIRust QP: {status:?}"));
182:                 Ok((lane_status(status), Vec::new()))
183:             }
184:         }
```

```text
346: fn settings((n, rank): Key) -> ReductionSettings {
347:     let mut s = ReductionSettings::new(n, rank);
348:     s.convention = Convention::MadLoop;
349:     s.tensor_weights = TensorWeights::Components;
350:     s.invariants = InvariantInput::SuppliedInvariants;
351:     s.scale_input = ScaleInput::MuSquared;
352:     s
353: }
354: fn dimensions(n: i32, rank: i32) -> Result<(Key, usize)> {
```

```text
490:         .map(|record| {
491:             let text = std::str::from_utf8(record)
492:                 .map_err(|_| "QP input is not ASCII")?
493:                 .trim();
494:             let mut value =
495:                 Float::parse(text, Some(113)).map_err(|e| format!("invalid QP value: {e}"))?;
496:             if !value.is_finite() {
497:                 return Err("nonfinite QP input".into());
498:             }
499:             value.set_prec(MP_BITS); // preserves signed zero and exact binary128 value
500:             Ok(value)
501:         })
502:         .collect()
503: }
504: fn mp_batch(
505:     n: usize,
```

## patched_executor

`.tirust-build/pole-development/crates/iregi-rust/src/direct/exec.rs`

SHA256 `255b250f6997cfd08f017a9962ac79e6a4a87421f1f2d9f7ddb6a8bc1129eace`

```text
404:         let stability_threshold = prototype.from_rational(&symbolica::domains::rational::Rational::from((
405:             symbolica::domains::integer::Integer::from(1),
406:             symbolica::domains::integer::Integer::from(1_000_000_000_000i64),
407:         )));
408:         Ok(Self {
409:             partial: None,
410:             external: Vec::new(),
411:             gram: None,
412:             gram_external: Vec::new(),
413:             gram_top: None,
414:             gram_alt_only: false,
415:             gram_alt_direct: false,
416:             gram_alternative_first: None,
417:             gram_tables: Vec::new(),
418:             gram_pool: gram::CoefficientPool::default(),
419:             gram_extra_ranks: 0,
420:             gram_wanted_ranks: None,
421:             gram_final: false,
422:             gram_everywhere: false,
423:             gram_estimated: false,
424:             estimate_flags: false,
425:             lane_mu2: None,
426:             shared: None,
```

```text
610:         a.recurrence += b.recurrence;
611:         a.gather += b.gather;
612:         a.output += b.output;
613:     }
614: 
615:     pub fn set_stability_threshold(&mut self, threshold: T) {
616:         self.stability_threshold = threshold;
617:     }
618: 
619:     pub fn set_input_precision(&mut self, bits: u32) {
620:         self.input_bits = bits;
621:         self.lightlike_tolerance = tolerance_at(&self.prototype, bits, LIGHTLIKE_MARGIN_BITS);
622:     }
623: 
624:     pub fn last_conditions(&self) -> &[Option<f64>] {
625:         &self.lane_conditions
626:     }
627: 
```

```text
1458:     /// Switch the lane to the plan variant of `relation_profile`.
1459:     fn select_variant(&mut self) -> Result<(), ReductionError> {
1460:         let key = (self.relation_profile.clone(), self.lightlike_profile.clone());
1461:         if self.relation_profile.is_empty() && self.lightlike_profile.is_empty() {
1462:             self.plan = self.base_plan.clone();
1463:             self.constants = self.base_constants.clone();
1464:         } else if let Some((plan, constants)) = self.variants.get(&key) {
1465:             self.plan = plan.clone();
1466:             self.constants = constants.clone();
1467:         } else {
1468:             let plan = Arc::new(DirectPlanData::with_profile_boost(
1469:                 self.base_plan.settings.clone(),
1470:                 &self.relation_profile,
1471:                 &self.lightlike_profile,
1472:                 self.base_plan.top_child_boost,
1473:             )?);
1474:             let constants: Arc<Vec<T>> = Arc::new(
1475:                 plan.constants
```

```text
1510:         self.gram_top_active = false;
1511:         self.estimate_tripped = false;
1512:         let n = plan.settings.propagators;
1513:         let threshold = self.stability_threshold.clone();
1514:         for relation in &plan.relations {
1515:             let weights = self
1516:                 .affine_relation(&relation.members)
1517:                 .map_err(GeometryFailure::Relation)?;
1518:             for (j, weight) in weights.into_iter().enumerate() {
1519:                 self.geometry[relation.weights as usize + j] = weight;
1520:             }
1521:         }
1522:         let top_mask = plan.subsets.last().map(|subset| subset.mask);
1523:         let partial = self.partial.is_some() && plan.jet.is_none() && plan.lambda_shift == 0;
1524:         let gram = self.gram.is_some() && plan.jet.is_none() && plan.lambda_shift == 0;
1525:         let flags = self.estimate_flags && plan.jet.is_none() && plan.lambda_shift == 0;
1526:         for (subset_id, subset) in plan.subsets.iter().enumerate() {
```

```text
1625:                                 let (mi, mj) = (members[i + 1], members[j + 1]);
1626:                                 self.matrix[i * size + j] =
1627:                                     csub(&cadd(d(origin, mi), d(origin, mj)), d(mi, mj));
1628:                                 self.rhs[i * size + j] =
1629:                                     if i == j { self.one.clone() } else { self.zero.clone() };
1630:                             }
1631:                         }
1632:                         for value in &self.matrix[..size * size] {
1633:                             let value = cnorm2(value).sqrt();
1634:                             if value > scale {
1635:                                 scale = value;
1636:                             }
1637:                         }
1638:                         let quality = solve_in_place(
1639:                             &mut self.matrix[..size * size],
1640:                             &mut self.rhs[..size * size],
1641:                             size,
1642:                             size,
1643:                         )
1644:                         .ok_or_else(|| {
1645:                             GeometryFailure::Singular(
1646:                                 subset.mask,
1647:                                 format!("singular Gram matrix for {members:?}"),
1648:                             )
1649:                         });
1650:                         if quality.is_err() {
1651:                             // Exactly singular pivot: conditioning zero.
1652:                             self.failed_condition = Some(0.0);
1653:                         }
1654:                         let quality = quality?;
1655:                         let power = subset.coefficient_rank as u64;
1656:                         // Hadamard-relative and scale-relative conditioning:
1657:                         // |det Z|^(1/n) against the recursion's kinematic scale.
1658:                         let mut scaled = quality.determinant.clone();
1659:                         for _ in 1..size {
1660:                             scaled = scaled.sqrt();
1661:                         }
1662:                         if size == 3 {
1663:                             scaled = quality.determinant.clone().powf(&(self.prototype.one() / &self.prototype.from_usize(3)));
1664:                         }
1665:                         let scaled = if scale.is_zero() { scale.clone() } else { scaled / &scale };
1666:                         let condition = if scaled < quality.relative_determinant {
1667:                             scaled
1668:                         } else {
1669:                             quality.relative_determinant.clone()
1670:                         };
1671:                         if condition.clone().pow(power) < threshold {
1672:                             if gram && (2..=3).contains(&size) {
1673:                                 // Small-Gram expansion (box or triangle), injected in the lane loop
1674:                                 // (with the a priori choice too: its retry covers refused lanes).
1675:                                 let estimate = condition.clone().pow(power).to_f64();
1676:                                 self.failed_condition =
1677:                                     Some(self.failed_condition.map_or(estimate, |f: f64| f.min(estimate)));
1678:                                 self.gram_external.push(subset_id);
1679:                                 continue;
1680:                             }
1681:                             if partial && (2..=3).contains(&size) && Some(subset.mask) != top_mask {
1682:                                 // Computed in double-double and injected.
1683:                                 let estimate = condition.clone().pow(power).to_f64();
1684:                                 self.failed_condition =
1685:                                     Some(self.failed_condition.map_or(estimate, |f: f64| f.min(estimate)));
1686:                                 self.external.push(subset_id);
1687:                                 continue;
1688:                             }
1689:                             self.failed_condition = Some(condition.clone().pow(power).to_f64());
1690:                             return Err(GeometryFailure::Singular(subset.mask, format!(
1691:                                 "ill-conditioned Gram matrix for {members:?}: conditioning {:e}",
1692:                                 condition
1693:                             )));
1694:                         }
1695:                         for j in 0..size {
```

```text
3439:             // Reuse the previous lane's variant; on a relation that does not
3440:             // hold here, restart from the base plan. Each affinely dependent
3441:             // subset the geometry discovers switches to a relation variant.
3442:             let mut failure = None;
3443:             // The lane's plan, cloned once (an atomic increment and
3444:             // decrement) and lent to every step below.
3445:             let plan = loop {
3446:                 let plan = self.plan.clone();
3447:                 match self.geometry(&plan) {
3448:                     Ok(()) => break plan,
3449:                     Err(GeometryFailure::Relation(error)) => {
3450:                         if self.relation_profile.is_empty() && self.lightlike_profile.is_empty() {
3451:                             failure = Some(error);
3452:                             break plan;
3453:                         }
3454:                         self.relation_profile.clear();
3455:                         self.lightlike_profile.clear();
3456:                         self.select_variant()?;
3457:                     }
3458:                     Err(GeometryFailure::Lightlike(mask)) => {
3459:                         if self.plan.jet.is_some() || self.lightlike_profile.contains(&mask) {
3460:                             failure = Some(format!(
3461:                                 "lightlike bubble {mask:#b} unsupported for dotted propagators"
3462:                             ));
3463:                             break plan;
3464:                         }
3465:                         self.lightlike_profile.push(mask);
3466:                         self.lightlike_profile.sort_unstable();
3467:                         self.select_variant()?;
3468:                     }
3469:                     Err(GeometryFailure::Unstable(error)) => {
3470:                         failure = Some(error);
3471:                         break plan;
3472:                     }
3473:                     Err(GeometryFailure::Singular(mask, error)) => {
3474:                         match self.relation_support(mask) {
3475:                             Ok(support)
3476:                                 if !self.relation_profile.contains(&support)
3477:                                     && !(batched_masters && support != 0) =>
3478:                             {
3479:                                 self.relation_profile.push(support);
3480:                                 self.relation_profile.sort_unstable();
3481:                                 self.select_variant()?;
3482:                             }
3483:                             _ => {
3484:                                 failure = Some(error);
3485:                                 break plan;
3486:                             }
3487:                         }
3488:                     }
3489:                 }
3490:             };
3491:             self.lane_estimate[row] = self.estimate_tripped;
3492:             if let Some(error) = failure {
3493:                 self.lane_conditions[row] = self.failed_condition;
```

## patched_plan

`.tirust-build/pole-development/crates/iregi-rust/src/direct/plan.rs`

SHA256 `f24b05ca3fbd74c68fde719da0efb6280d341b0a75e6eee7cc9ca42711626b2a`

```text
230: /// Highest pentagon coefficient rank from the Gram-free Denner--Dittmaier
231: /// reduction (hep-ph/0509141, sect. 5, with the rank-5 UV remainder); higher
232: /// pentagon ranks continue with the four-dimensional Gram recursion.
233: pub(crate) const DD_PENTAGON_RANK: i32 = 5;
234: 
```

```text
698:     pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
699:         Self::with_relations(settings, &[])
700:     }
701: 
702:     /// Plan variant in which every subset (of at most five propagators)
703:     /// containing one of `relation_masks` is reduced by that subset's
704:     /// constant-one denominator identity instead of its singular Gram system.
705:     pub fn with_relations(
706:         settings: ReductionSettings,
707:         relation_masks: &[SubsetMask],
708:     ) -> Result<Self, ReductionError> {
709:         Self::with_profile(settings, relation_masks, &[])
710:     }
711: 
712:     /// Plan variant with denominator identities for `relation_masks` and
713:     /// closed-form coefficients for the lightlike bubbles `lightlike_masks`.
714:     pub fn with_profile(
715:         settings: ReductionSettings,
716:         relation_masks: &[SubsetMask],
717:         lightlike_masks: &[SubsetMask],
718:     ) -> Result<Self, ReductionError> {
719:         Self::with_profile_boost(settings, relation_masks, lightlike_masks, 0)
720:     }
721: 
722:     /// Plan whose top subset's pinches carry `boost` more coefficient ranks
723:     /// than the top's recursion needs: the small-Gram expansion of a box
724:     /// reaches its rank-`r` coefficients from triangles of rank `r + boost`.
725:     pub fn with_top_child_boost(settings: ReductionSettings, boost: u16) -> Result<Self, ReductionError> {
726:         Self::with_profile_boost(settings, &[], &[], boost)
727:     }
728: 
729:     pub fn with_profile_boost(
730:         settings: ReductionSettings,
731:         relation_masks: &[SubsetMask],
732:         lightlike_masks: &[SubsetMask],
733:         boost: u16,
734:     ) -> Result<Self, ReductionError> {
735:         Self::supported(&settings)?;
```

```text
749:         let rank = settings.rank;
750:         let basis = tensor_basis(rank);
751:         let monomials = MonomialTable::new(rank);
752:         let top = full_mask(n);
753:         // Gram-free pentagon rank: not for dotted propagators (no geometry
754:         // jets of X^-1 yet) or when a box child is degenerate (it supplies
755:         // moments by its identity, not PV coefficients).
756:         let dd_rank = |mask: SubsetMask, moment: i32| -> i32 {
757:             if moment < 0 || jet.is_some() {
758:                 return -1;
759:             }
760:             if members(mask).into_iter().any(|member| relation_for(without(mask, member)).is_some()) {
761:                 return -1;
762:             }
763:             moment.min(DD_PENTAGON_RANK)
764:         };
765: 
766:         // Propagate rank needs from the top topology to its pinches.
767:         let mut coefficient_need: HashMap<SubsetMask, i32> = HashMap::new();
768:         let mut moment_need: HashMap<SubsetMask, i32> = HashMap::new();
769:         moment_need.insert(top, i32::from(rank));
770:         // Every nonempty subset: memory, not a cap, bounds the point count.
771:         let mut masks: Vec<SubsetMask> = Vec::new();
772:         masks.try_reserve_exact(top as usize).map_err(|_| {
773:             ReductionError::Unsupported(format!(
774:                 "direct engine: the 2^{n} - 1 subsets of a {n}-propagator plan exceed the available memory"
775:             ))
776:         })?;
777:         masks.extend(1..=top);
778:         masks.sort_by_key(|m| std::cmp::Reverse(m.count_ones()));
779:         for &mask in &masks {
780:             let size = mask.count_ones() as usize;
781:             let moment = moment_need.get(&mask).copied().unwrap_or(-1);
782:             if let Some(relation) = relation_for(mask) {
783:                 if moment >= 0 {
784:                     for member in members(relation_masks[relation]) {
785:                         let child = without(mask, member);
786:                         let need = moment_need.entry(child).or_insert(-1);
787:                         *need = (*need).max(moment);
788:                     }
789:                 }
790:                 continue;
791:             }
792:             match size {
793:                 5.. => {
794:                     let dd = if size == 5 { dd_rank(mask, moment) } else { -1 };
795:                     if moment > dd {
796:                         for member in members(mask) {
797:                             let child = without(mask, member);
798:                             let need = moment_need.entry(child).or_insert(-1);
799:                             *need = (*need).max((moment - 1).max(0));
800:                         }
801:                     }
802:                     if dd >= 0 {
803:                         // Box coefficients up to the same rank (Denner-Dittmaier).
804:                         for member in members(mask) {
805:                             let child = without(mask, member);
806:                             let need = coefficient_need.entry(child).or_insert(-1);
807:                             *need = (*need).max(dd);
808:                         }
809:                     }
810:                 }
811:                 _ => {
812:                     if moment >= 0 {
813:                         let need = coefficient_need.entry(mask).or_insert(-1);
814:                         *need = (*need).max(moment + 2 * i32::from(lambda_shift));
815:                     }
816:                     let coefficient = coefficient_need.get(&mask).copied().unwrap_or(-1);
817:                     if coefficient >= 1 && size >= 2 {
818:                         let extra = if mask == top { i32::from(boost) } else { 0 };
819:                         for member in members(mask) {
820:                             let child = without(mask, member);
821:                             let need = coefficient_need.entry(child).or_insert(-1);
```

```text
879:                 members: member_list.clone(),
880:             });
881:             geometry_count = grow(geometry_count, member_list.len() as u32)?;
882:         }
883:         for &mask in &masks {
884:             let coefficient = coefficient_need.get(&mask).copied().unwrap_or(-1);
885:             let moment = moment_need.get(&mask).copied().unwrap_or(-1);
886:             if coefficient < 0 && moment < 0 {
887:                 continue;
888:             }
889:             let member_list = members(mask);
890:             let size = member_list.len();
891:             let kind = match (relation_for(mask), size) {
892:                 (Some(relation), _) => SubsetKind::Relation(relation),
893:                 (None, 6..) => SubsetKind::Hexagon,
894:                 (None, 5) => SubsetKind::Pentagon,
895:                 _ => SubsetKind::Pv,
896:             };
897:             let geometry_base = geometry_count;
898:             let mut subset = Subset {
899:                 mask,
900:                 members: member_list,
901:                 kind,
902:                 coefficient_rank: coefficient,
903:                 moment_rank: moment,
904:                 geometry_base,
905:                 pv: None,
906:                 pentagon: None,
907:                 hexagon: None,
908:                 master: None,
909:                 moment_base: None,
```

## patched_numeric

`.tirust-build/pole-development/crates/iregi-rust/src/direct/numeric.rs`

SHA256 `c4189f1e59f0746ea4413319b9a85df926789c3b1c95f9e1461bf7088879ebee`

```text
269: /// Outcome of a small dense solve.
270: pub(crate) struct SolveQuality<T> {
271:     /// `|det(A)| / prod_i ||row_i||_2`, at most one; zero for singular input.
272:     pub relative_determinant: T,
273:     /// `|det(A)|`.
274:     pub determinant: T,
275: }
276: 
277: /// Solve `A X = B` in place by Gaussian elimination with partial pivoting.
278: /// `a` is `n x n` row-major and is destroyed; `b` is `n x m` row-major and
279: /// receives `X`. Returns `None` for an exactly singular pivot.
280: pub(crate) fn solve_in_place<T: NativeFloat>(
281:     a: &mut [C<T>],
282:     b: &mut [C<T>],
283:     n: usize,
284:     m: usize,
285: ) -> Option<SolveQuality<T>> {
286:     debug_assert_eq!(a.len(), n * n);
287:     debug_assert_eq!(b.len(), n * m);
288:     let one = a[0].re.one();
289:     // Hadamard bound for the relative determinant estimate.
290:     let mut hadamard = one.clone();
```

```text
341:             for j in column + 1..n {
342:                 let update = cmul(&a[column * n + j], &b[j * m + k]);
343:                 value = csub(&value, &update);
344:             }
345:             b[column * m + k] = cmul(&value, &inverse);
346:         }
347:     }
348:     let relative_determinant = if hadamard.is_zero() {
349:         hadamard
350:     } else {
351:         determinant_norm.clone() / &hadamard
352:     };
353:     Some(SolveQuality {
354:         relative_determinant,
```

## patched_direct_api

`.tirust-build/pole-development/crates/iregi-rust/src/direct.rs`

SHA256 `71f78f5b1ed56890371a5adb9081753e2a28306847957bf1a315e39189689b46`

```text
188:     /// Workspace with the default direct oneloopmaster master kernels.
189:     pub fn workspace<T: NativeFloat>(&self, prototype: T) -> Result<DirectWorkspace<T>, ReductionError>
190:     where
191:         Complex<T>: EvaluationDomain,
192:     {
193:         self.workspace_with_masters(prototype, DirectMasters::Native)
194:     }
195: 
196:     /// Workspace with batched `MasterEvaluator` masters under `policy`.
197:     pub fn workspace_with_policy<T: NativeFloat>(
198:         &self,
199:         prototype: T,
200:         policy: MasterExecutionPolicy,
201:     ) -> Result<DirectWorkspace<T>, ReductionError>
202:     where
203:         Complex<T>: EvaluationDomain,
204:     {
205:         self.workspace_with_masters(prototype, DirectMasters::Policy(policy))
206:     }
207: 
208:     pub fn workspace_with_masters<T: NativeFloat>(
209:         &self,
210:         prototype: T,
211:         masters: DirectMasters,
212:     ) -> Result<DirectWorkspace<T>, ReductionError>
213:     where
214:         Complex<T>: EvaluationDomain,
215:     {
216:         Ok(DirectWorkspace {
217:             executor: DirectExecutor::new(self.data.clone(), prototype, masters)?,
218:         })
219:     }
220: }
```

## patched_types

`.tirust-build/pole-development/crates/iregi-rust/src/types.rs`

SHA256 `a29dc5f68edae92c5f35acc49ac8b2c64dbc8a38dd078df98d13769a801103e0`

```text
175:     pub fn new(propagators: usize, rank: u16) -> Self {
176:         Self {
177:             propagators,
178:             rank,
179:             powers: vec![1; propagators],
180:             dimension_shift: 0,
181:             algorithm: Algorithm::Hybrid,
182:             convention: Convention::IregiNative,
183:             tensor_weights: TensorWeights::Components,
184:             invariants: InvariantInput::FromShifts,
185:             input_policy: InputPolicy::Preserve,
186:             scale_input: ScaleInput::Mu,
187:         }
188:     }
189: }
190: 
191: #[derive(Debug, thiserror::Error)]
192: pub enum ReductionError {
193:     #[error("invalid input: {0}")]
194:     InvalidInput(String),
195:     #[error("reduction preparation: {0}")]
```
