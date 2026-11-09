## historical_bridge

Path: `.tirust-build/bridge-poles/src/lib.rs`

SHA256: `1eab09d61652e4dfade900a22f8c63a071b4fecee2e7454e8144f934f17e34e0`

```text
21: pub const ABI_VERSION: u32 = 1;
22: const SUCCESS: i32 = 0;
23: const UNSTABLE: i32 = 1;
24: const ERROR: i32 = 2;
25: const UNSUPPORTED: i32 = 3;
26: const RECORD: usize = 64;
27: const MP_BITS: u32 = 256;
28: const DEFAULT_BUDGET: u64 = 268_435_456;
29: type Key = (usize, u16);
30: type Result<T> = std::result::Result<T, String>;
```

```text
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
```

```text
238:     #[cfg(feature = "poles-only")]
239:     fn mp_poles(
240:         &mut self,
241:         key: Key,
242:         batch: KinematicsBatch<Float>,
243:         mode: Mode,
244:     ) -> Result<(i32, Vec<u8>)> {
245:         if !self.pole_mp.contains_key(&key) {
246:             let mut workspace = self.plans[&key]
247:                 .pole_workspace(Float::new(MP_BITS))
248:                 .map_err(|e| e.to_string())?;
249:             workspace.set_input_precision(113);
250:             workspace.set_stability_threshold(Float::with_val(MP_BITS, 2).pow(136).inv());
251:             self.pole_mp.insert(key, workspace);
252:         }
253:         let result = self
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

## current_bridge

Path: `vendor/tirust/src/lib.rs`

SHA256: `455529b929b725e8fe2d7337ab4a41fa3de9dee1b3a953afe6dc032cf99e5a4c`

```text
171:             return self.mp_poles(key, batch, mode);
172:             #[cfg(not(feature = "poles-only"))]
173:             return Ok((UNSUPPORTED, Vec::new()));
174:         }
175:         if !self.mp.contains_key(&key) {
176:             let mut workspace = self.plans[&key]
177:                 .workspace(Float::new(MP_BITS))
178:                 .map_err(|e| e.to_string())?;
179:             workspace.set_input_precision(113);
180:             // 256-bit arithmetic with substantial headroom above binary128.
181:             // The external MadLoop check still decides numerical acceptance.
182:             workspace.set_stability_threshold(Float::with_val(MP_BITS, 2).pow(136).inv());
183:             self.mp.insert(key, workspace);
184:         }
```

```text
252:         batch: KinematicsBatch<Float>,
253:         mode: Mode,
254:     ) -> Result<(i32, Vec<u8>)> {
255:         if !self.pole_mp.contains_key(&key) {
256:             let mut workspace = self.plans[&key]
257:                 .pole_workspace(Float::new(MP_BITS))
258:                 .map_err(|e| e.to_string())?;
259:             workspace.set_input_precision(113);
260:             workspace.set_stability_threshold(Float::with_val(MP_BITS, 2).pow(136).inv());
261:             self.pole_mp.insert(key, workspace);
262:         }
263:         let result = self
```

```text
336: }
337: fn qp_cache_key(key: Key, records: [&[u8]; 4], mode: Mode) -> Vec<u8> {
338:     let mut bytes = Vec::with_capacity(16 + records.iter().map(|x| x.len()).sum::<usize>());
339:     bytes.extend_from_slice(&(key.0 as u32).to_le_bytes());
340:     bytes.extend_from_slice(&key.1.to_le_bytes());
341:     bytes.push(mode.poles as u8);
342:     bytes.push(u8::from(mode.finite));
343:     bytes.extend_from_slice(&113u32.to_le_bytes());
344:     bytes.extend_from_slice(&MP_BITS.to_le_bytes());
345:     for record in records {
346:         bytes.extend_from_slice(record);
347:     }
348:     bytes
349: }
```

```text
493:             }
494:         }
495:     }
496:     out
497: }
498: fn parse_records(data: &[u8]) -> Result<Vec<Float>> {
499:     data.chunks_exact(RECORD)
500:         .map(|record| {
501:             let text = std::str::from_utf8(record)
502:                 .map_err(|_| "QP input is not ASCII")?
503:                 .trim();
504:             let mut value =
505:                 Float::parse(text, Some(113)).map_err(|e| format!("invalid QP value: {e}"))?;
506:             if !value.is_finite() {
507:                 return Err("nonfinite QP input".into());
508:             }
509:             value.set_prec(MP_BITS); // preserves signed zero and exact binary128 value
510:             Ok(value)
511:         })
```

## patched_plan

Path: `.tirust-build/pole-development/crates/iregi-rust/src/direct/plan.rs`

SHA256: `f24b05ca3fbd74c68fde719da0efb6280d341b0a75e6eee7cc9ca42711626b2a`

```text
230: /// Highest pentagon coefficient rank from the Gram-free Denner--Dittmaier
231: /// reduction (hep-ph/0509141, sect. 5, with the rank-5 UV remainder); higher
232: /// pentagon ranks continue with the four-dimensional Gram recursion.
233: pub(crate) const DD_PENTAGON_RANK: i32 = 5;
234: 
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
```

```text
956:                     if dd >= 0 {
957:                         // Momenta and origin for the gather, X^-1, Xtt/det X.
958:                         let (geometry, length) = PvGeometry::new(geometry_count, 4);
959:                         subset.pv = Some(geometry);
960:                         geometry_count = grow(geometry_count, length)?;
961:                         pentagon.xinv = geometry_count;
962:                         geometry_count = grow(geometry_count, 25)?;
963:                         pentagon.adjadj = geometry_count;
964:                         geometry_count = grow(geometry_count, 64)?;
965:                         pentagon.sums = geometry_count;
966:                         geometry_count = grow(geometry_count, 21)?;
967:                         subset.coefficient_rank = dd;
968:                         let mut layout = CoefficientLayout {
969:                             lookup: FastMap::default(),
```

## patched_executor

Path: `.tirust-build/pole-development/crates/iregi-rust/src/direct/exec.rs`

SHA256: `255b250f6997cfd08f017a9962ac79e6a4a87421f1f2d9f7ddb6a8bc1129eace`

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
1696:                             let mut zs = self.zero.clone();
1697:                             let mut zf = self.zero.clone();
1698:                             for i in 0..size {
1699:                                 let value = self.rhs[j * size + i].clone();
```

```text
1720:     #[inline(never)]
1721:     fn geometry_pentagon(&mut self, subset: &Subset, n: usize) -> Result<(), GeometryFailure> {
1722:         let threshold = self.stability_threshold.clone();
1723:         let pentagon_calibration = self.binary_tolerance(7);
1724:         let members = &subset.members;
1725:         let origin = members[0];
1726:         let d = |i: usize, j: usize| &self.distances[i * n + j];
1727:         let geometry = subset.pentagon.unwrap();
1728:         // b = R^-1 1 for the scalar pentagon.
1729:         for i in 0..5 {
1730:             for j in 0..5 {
1731:                 let (mi, mj) = (members[i], members[j]);
1732:                 self.matrix[i * 5 + j] = csub(
1733:                     &csub(d(mi, mj), &self.masses[mi]),
1734:                     &self.masses[mj],
1735:                 );
1736:             }
1737:             self.rhs[i] = self.one.clone();
1738:         }
1739:         let quality = solve_in_place(&mut self.matrix[..25], &mut self.rhs[..5], 5, 1)
1740:             .ok_or_else(|| GeometryFailure::Unstable(format!("singular Cayley matrix for {members:?}")));
1741:         if quality.is_err() {
1742:             // Exactly singular pivot: conditioning zero.
1743:             self.failed_condition = Some(0.0);
1744:         }
1745:         let quality = quality?;
1746:         if quality.relative_determinant < threshold {
1747:             self.failed_condition = Some(quality.relative_determinant.to_f64());
1748:             return Err(GeometryFailure::Unstable(format!(
1749:                 "ill-conditioned Cayley matrix for {members:?}: relative determinant {:e}",
1750:                 quality.relative_determinant
1751:             )));
1752:         }
1753:         for i in 0..5 {
1754:             self.geometry[geometry.b as usize + i] = self.rhs[i].clone();
1755:         }
1756:         if geometry.dd_rank >= 1 {
1757:             // Gram-free coefficients: momenta and origin for the
1758:             // gather, X^-1 of X = [[2 m0^2, f^T], [f, Z]] and the
1759:             // second adjugates Xtt_(kn)(0i) / det X.
1760:             let pv = subset.pv.unwrap();
1761:             for i in 0..4 {
1762:                 let member = members[i + 1];
1763:                 let f = cadd(&csub(d(origin, member), &self.masses[member]), &self.masses[origin]);
1764:                 self.matrix[i + 1] = f.clone();
1765:                 self.matrix[(i + 1) * 5] = f;
1766:                 for j in 0..4 {
1767:                     let (mi, mj) = (members[i + 1], members[j + 1]);
1768:                     self.matrix[(i + 1) * 5 + j + 1] =
1769:                         csub(&cadd(d(origin, mi), d(origin, mj)), d(mi, mj));
1770:                 }
1771:                 for a in 0..4 {
1772:                     self.geometry[pv.p as usize + 4 * i + a] =
1773:                         csub(&self.shifts[member][a], &self.shifts[origin][a]);
1774:                 }
1775:             }
1776:             for a in 0..4 {
1777:                 self.geometry[pv.origin as usize + a] = self.shifts[origin][a].clone();
1778:             }
1779:             self.matrix[0] = cadd(&self.masses[origin], &self.masses[origin]);
1780:             for i in 0..5 {
1781:                 for j in 0..5 {
1782:                     self.rhs[i * 5 + j] = if i == j { self.one.clone() } else { self.zero.clone() };
1783:                 }
1784:             }
1785:             let quality = solve_in_place(&mut self.matrix[..25], &mut self.rhs[..25], 5, 5)
1786:                 .ok_or_else(|| GeometryFailure::Unstable(format!("singular modified Cayley matrix for {members:?}")));
1787:             if quality.is_err() {
1788:                 self.failed_condition = Some(0.0);
1789:             }
1790:             let quality = quality?;
1791:             if quality.relative_determinant < threshold {
1792:                 self.failed_condition = Some(quality.relative_determinant.to_f64());
1793:                 return Err(GeometryFailure::Unstable(format!(
1794:                     "ill-conditioned modified Cayley matrix for {members:?}: relative determinant {:e}",
1795:                     quality.relative_determinant
1796:                 )));
1797:             }
1798:             let xinv: Vec<C<T>> = self.rhs[..25].to_vec();
1799:             for (i, value) in xinv.iter().enumerate() {
1800:                 self.geometry[geometry.xinv as usize + i] = value.clone();
1801:             }
1802:             let sums = geometry.sums as usize;
1803:             let mut column = self.zero.clone();
1804:             for m in 1..=4 {
```

```text
1857:                 .ok_or_else(|| GeometryFailure::Singular(subset.mask, format!("singular pentagon Gram matrix for {members:?}")));
1858:             if quality.is_err() {
1859:                 // Exactly singular pivot: conditioning zero.
1860:                 self.failed_condition = Some(0.0);
1861:             }
1862:             let _ = quality?;
1863:             // The moment recursion applies Z^-1 once per rank, so
1864:             // rounding grows like kappa^r with kappa the condition
1865:             // number ||Z|| ||Z^-1|| (row sums; rhs now holds Z^-1).
1866:             // On physical 2->3 points the binary64 error was within
1867:             // ~50 kappa^r u (u = 2^-53); the Hadamard-relative
1868:             // determinant missed this (accepted lanes with E/A
1869:             // up to 3e-4), though kappa alone also overestimates
1870:             // (an opening N6 row with kappa 60 is good to 2e-11).
1871:             // This gate only removes hopeless lanes: 2^-7/kappa^r
1872:             // against the threshold accepts kappa^r <= 8.6e9 in
1873:             // binary64 (1e-12) and kappa^r <= 2^(bits-47) at a
1874:             // rescue tier. The binary64 hybrid checks the
1875:             // remaining N>=5 lanes a posteriori (loop rotation).
1876:             let mut inverse_norm = self.prototype.zero();
1877:             for i in 0..4 {
1878:                 let mut row = self.prototype.zero();
1879:                 for j in 0..4 {
1880:                     row += cnorm2(&self.rhs[i * 4 + j]).sqrt();
1881:                 }
1882:                 if row > inverse_norm {
1883:                     inverse_norm = row;
1884:                 }
1885:             }
1886:             let kappa = norm * &inverse_norm;
1887:             let condition = if kappa.is_zero() || !kappa.is_finite() {
1888:                 self.prototype.zero()
1889:             } else {
1890:                 pentagon_calibration.clone()
1891:                     / &kappa.clone().pow((subset.moment_rank - geometry.dd_rank.max(0)) as u64)
1892:             };
1893:             if condition < threshold {
1894:                 self.failed_condition = Some(condition.to_f64());
1895:                 return Err(GeometryFailure::Singular(subset.mask, format!(
1896:                     "ill-conditioned pentagon Gram matrix for {members:?}: condition number {:e}",
1897:                     kappa
1898:                 )));
1899:             }
1900:             for mu in 0..4 {
1901:                 let mut a = csub(&self.zero, &self.shifts[origin][mu]);
```

```text
1981:             if best.as_ref().is_none_or(|(relative, _, _)| quality.relative_determinant > *relative) {
1982:                 best = Some((quality.relative_determinant, self.rhs[..25].to_vec(), drop));
1983:             }
1984:         }
1985:         let Some((relative, solution, drop)) = best else {
1986:             // Exactly singular pivot: conditioning zero.
1987:             self.failed_condition = Some(0.0);
1988:             return Err(GeometryFailure::Unstable(format!(
1989:                 "singular six-point coordinate system for {members:?}"
1990:             )));
1991:         };
1992:         if relative < threshold {
1993:             self.failed_condition = Some(relative.to_f64());
1994:             return Err(GeometryFailure::Unstable(format!(
1995:                 "ill-conditioned six-point coordinate system: relative determinant {:e}",
1996:                 relative
1997:             )));
1998:         }
1999:         let chosen: Vec<usize> = (0..size).filter(|&k| drop & (1 << k) == 0).collect();
2000:         for column in 0..5 {
2001:             let mut sum = self.zero.clone();
2002:             for i in 0..5 {
2003:                 let value = solution[i * 5 + column].clone();
2004:                 sum = cadd(&sum, &value);
2005:                 self.geometry[weights + size * column + chosen[i + 1]] = value;
```

## patched_numeric

Path: `.tirust-build/pole-development/crates/iregi-rust/src/direct/numeric.rs`

SHA256: `c4189f1e59f0746ea4413319b9a85df926789c3b1c95f9e1461bf7088879ebee`

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

## upstream_direct_api

Path: `../references/TIRust/crates/iregi-rust/src/direct.rs`

SHA256: `02c834e314833392b7fe9eaafa6cb106e1273bfdccd5b4c9105ddf76dd82b61b`

```text
430:     pub fn set_stability_threshold(&mut self, threshold: T) {
431:         self.executor.set_stability_threshold(threshold);
432:     }
433: 
434:     /// Precision (bits) of the inputs this workspace evaluates; by default
435:     /// its working precision. A bubble is lightlike (closed-form coefficients)
436:     /// when |p^2| is at most 2^-(bits - 13) of its scale: 2^-40 for binary64
437:     /// inputs, also when they are evaluated at a higher precision (rescue
438:     /// tiers), and relative to the working precision for inputs given at it.
439:     pub fn set_input_precision(&mut self, bits: u32) {
440:         self.executor.set_input_precision(bits);
441:     }
442: 
443:     /// Per lane of the most recent call: for lanes refused on conditioning,
444:     /// the refused estimate (the quantity compared with the stability
445:     /// threshold; zero for an exactly singular pivot), else `None`. Callers
446:     /// use it to choose a working precision for a retry.
447:     pub fn last_conditions(&self) -> &[Option<f64>] {
448:         self.executor.last_conditions()
449:     }
```

## upstream_precision_ladder

Path: `../references/TIRust/crates/iregi-rust/src/dispatch/direct_hybrid.rs`

SHA256: `230cf32dab679c6f64f44056859bbc43ff8ff764a78f6d6ad9165c0082166b28`

```text
117: /// Working precisions of the direct-engine rescue: double-double, then MPFR.
118: pub(super) const RESCUE_BITS: [u32; 3] = [106, 256, 512];
119: /// A tier accepts conditioning down to `2^-(bits - RESCUE_HEADROOM)`, so the
120: /// estimated amplification of its rounding error stays below `2^-40`
121: /// (stricter than the binary64 gate of `1e-12` at 53 bits).
122: pub(super) const RESCUE_HEADROOM: u32 = 40;
123: /// Bits added to the binary64 estimate when choosing the first tier.
124: pub(super) const RESCUE_MARGIN: u32 = 8;
125: 
126: /// Largest accepted E/A between a binary64 lane and its rotated copy.
127: const ROTATION_TOLERANCE: f64 = 1.0 / (1u64 << 34) as f64;
128: /// Binary64 conditioning gate of plans with the loop-rotation test (N >= 4,
129: /// uniform powers). The test checks every accepted lane a posteriori, so the
130: /// a priori gate only removes hopeless lanes: on physical gg->Hgg pentagons
131: /// and gg->gggg hexagons (stability-20261030) the default `1e-12` refused
132: /// lanes of which most were good to 1e-10. The gate alone bounds nothing:
133: /// in stability capture 044 (`RotationTest::Disabled`, against 256 bits) the
134: /// lanes it accepts reach E/A 7.0e-6 for rank-4 boxes, 1.6e-7 for rank-6
135: /// boxes, 1.2e-9 for pentagons, 7.0e-10 for hexagons and 3.7e-9 for
136: /// heptagons; after the rotation test the same points stay within 6.7e-11.
137: pub(super) const ROTATION_GATE: f64 = 1e-15;
138: 
139: /// Gate of the checked binary64 retry. On the stability-20261030 points
140: /// (boxes to heptagons) no lane below 1e-21 passed the rotation check.
141: const CHECKED_GATE: f64 = 1e-30;
142: /// Bits by which a lane below the a priori gate must beat the rotation
143: /// tolerance. Its rotated copy shares the ill-conditioned subsystems, so the
144: /// difference underestimates the error more often: on stability-20261030
145: /// such lanes accepted at the plain tolerance reached E/A 6.6 times it, with
```

```text
282: /// Headroom of the double-double tier. Its acceptance is checked a
283: /// posteriori by the loop-rotation test for N >= 4 (`RESCUE_ROTATION_TOLERANCE`):
284: /// on physical gg -> ggggg heptagons the conditioning estimate called for 256
285: /// bits, while the 106-bit direct engine with no gate at all agreed with 256
286: /// bits to E/A <= 1.7e-23 (30 points).
287: const RESCUE_DD_HEADROOM: u32 = 20;
288: /// Largest accepted E/A between a 106-bit lane and its rotated copy, both
289: /// rounded to binary64 (a few ulps).
290: const RESCUE_ROTATION_TOLERANCE: f64 = 1.0 / (1u64 << 46) as f64;
291: /// Binary64 conditioning estimate (2^-53 over the rotation difference) at or
292: /// above which a lane rescued at double-double skips that tier's rotation
293: /// check: its binary64 difference was at most 2^-20, so the double-double
294: /// error is ~2^-73, far below `RESCUE_ROTATION_TOLERANCE` (the opening N7R7
295: /// batch of 32 spent half of its time in that check).
296: const RESCUE_CHECK_CONDITION: f64 = 1.0 / (1u64 << 33) as f64;
297: 
298: fn rescue_threshold(bits: u32) -> f64 {
299:     let headroom = if bits == RESCUE_BITS[0] { RESCUE_DD_HEADROOM } else { RESCUE_HEADROOM };
300:     2f64.powi(-((bits - headroom) as i32))
301: }
302: 
303: /// First tier that admits the binary64 conditioning estimate with margin. An
304: /// exactly singular binary64 pivot carries no estimate and starts at 256 bits.
305: fn first_tier(condition: f64) -> usize {
306:     first_tier_under(condition, rescue_threshold)
307: }
308: 
309: /// [`first_tier`] for tiers that accept conditioning down to `threshold(bits)`
310: /// (the facade's rescue or embedder mode's stricter tiers).
311: pub(super) fn first_tier_under(condition: f64, threshold: impl Fn(u32) -> f64) -> usize {
312:     if condition > 0.0 {
313:         RESCUE_BITS
314:             .iter()
315:             .position(|&bits| condition >= threshold(bits) * 2f64.powi(RESCUE_MARGIN as i32))
316:             .unwrap_or(RESCUE_BITS.len() - 1)
317:     } else {
318:         1
319:     }
```

## upstream_embedded

Path: `../references/TIRust/crates/iregi-rust/src/dispatch/embedded.rs`

SHA256: `c4e392171b243c408c4a2802f30e3a90938c2c6cfc70c85d994ff426cce201c4`

```text
96: // The facade's own constants: the method choice runs with its gate and
97: // small-Gram boost, the precision tiers on its rescue ladder.
98: use super::direct_hybrid::{
99:     GRAM_BOOST, RESCUE_BITS as TIER_BITS, RESCUE_HEADROOM as TIER_HEADROOM,
100:     ROTATION_GATE as METHOD_CHOICE_GATE, from_binary64,
101: };
102: 
103: /// A tier accepts conditioning down to `2^-(bits - TIER_HEADROOM)`: its
104: /// rounding error amplified by at most `2^-40` relative, below binary64's
105: /// accuracy (no a posteriori check in this mode).
106: fn tier_threshold(bits: u32) -> f64 {
107:     2f64.powi(-((bits - TIER_HEADROOM) as i32))
108: }
109: 
```

## upstream_reference_test

Path: `../references/TIRust/crates/iregi-rust/tests/lifted_caps.rs`

SHA256: `f1e043bcae2e2c86eee79392586c07077952ae9a8229f208fe43d88c91e019e1`

```text
497:     let bits = reference.get_precision();
498:     let mut workspace = plan.workspace(reference.clone()).unwrap();
499:     // Conditioning accepted down to 2^-(bits - 56) (2^-200 at 256 bits): the
500:     // reference keeps far more than binary64's digits, the rotation checks.
501:     workspace.set_stability_threshold(lift(&reference, 2f64.powi(-((bits - 56) as i32))));
502:     let lifted = |values: Vec<Complex<f64>>| -> Vec<Complex<R>> {
503:         values.iter().map(|z| Complex::new(lift(&reference, z.re), lift(&reference, z.im))).collect()
504:     };
505:     let start = std::time::Instant::now();
506:     let precise = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lifted(values) }).unwrap();
507:     let evaluated_reference = start.elapsed();
508:     assert_eq!(precise.statuses[0], LaneStatus::Success, "{bits} bits");
509:     let other = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lifted(row(&rotated)) }).unwrap();
510:     assert_eq!(other.statuses[0], LaneStatus::Success, "{bits} bits, rotated");
511:     let order = precise_deviation(other.lane(0).unwrap(), precise.lane(0).unwrap(), 2 * bits);
512:     let achieved = match binary64.lane(0) {
513:         Ok(lane) => format!("{:e}", relative_deviation(lane, precise.lane(0).unwrap())),
```
