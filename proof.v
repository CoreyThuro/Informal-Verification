Require Import ZArith.

Theorem sum_even: forall a b: Z, (exists k: Z, a = 2 * k) -> (exists m: Z, b = 2 * m) -> (exists n: Z, a + b = 2 * n).
Proof.
  intros a b [k Hk] [m Hm].
  exists (k + m).
  rewrite Hk, Hm.
  ring.
Qed.