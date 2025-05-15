<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
$$
\bm{X} \in \mathbb{R}^{B \times T \times C}, \quad 
\bm{W}_Q, \bm{W}_K, \bm{W}_V, \bm{W}_E \in \mathbb{R}^{C \times d}
$$
$$
D = \frac{C}{H}, \quad 
\bm{M} \in \{0, -\infty\}^{T \times T}, \quad 
\bm{W}_O \in \mathbb{R}^{C \times C}, \quad 
\text{ReLU}, \quad 
\bm{W}_G \in \mathbb{R}^{C \times F}, \quad 
\bm{W}_2 \in \mathbb{R}^{F \times C}
$$
$$
\bm{Q}_n = \bm{X} \bm{W}_Q^{(n)}, \quad 
\bm{K}_n = \bm{X} \bm{W}_K^{(n)}, \quad 
\bm{V}_h = \bm{X} \bm{W}_V^{(n)}
$$

$$
\bm{A}_h = \text{Softmax}\left(\frac{\bm{Q}_n \bm{K}_n^\top}{\sqrt{d}} + \bm{M} \right), \quad 
\bm{O}_h = \bm{A}_h \bm{V}_h
$$

$$
\text{MHSA}(\bm{X}) = \text{Concat}(\bm{O}_1, \dots, \bm{O}_H) \bm{W}_O, \quad 
\bm{X}' = \bm{X} + \text{MH}
$$

$$
\bm{X}'' = \bm{X}' + \bm{W}_2 \left( \text{ReLU}\left( \bm{W}_1 (\text{LayerNorm}(\bm{X}')) \right) \right)
$$

$$
\text{SA}(\text{LayerNorm}(\bm{X})), \quad 
\text{Logits} = \bm{X}'' \bm{W}_{\text{out}} + \bm{B}_{\text{out}}

$$
# HGBERT-developers
HGBERT is a High Generative Bidirectional Encoder AI. Which is under no ones license
The Code is written from scratch in Python3 using pytorch, Does not use copy pasted code from Any Premade models.
Inspired by OpenAI's GPT and Google's BERT.
# Coders
Vukasin Trujkic
Hans Pilgaard Winther
Morris.
# PAGE
HGBERT.free.nf
