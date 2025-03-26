Require Import Arith.
Require Import Lia.

Theorem example: forall n: nat, P(n).
Proof.
  induction n.
  
  (* Base case: n = 0 *)
  simpl.
  auto.
  
  (* Inductive step: n = S n *)
  simpl.
  rewrite IHn.
  auto.
Qed.