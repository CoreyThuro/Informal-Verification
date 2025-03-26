Require Import Arith.
Require Import Lia.

Theorem example: forall n: nat, exists k: nat, n + n = 2 * k.
Proof.
  intros n.
  exists n.
  ring.
Qed.