# Visualizing small Attention-only Transformers

*Work done during an internship at MILES, Paris Dauphine University, under the supervision of Yann Chevaleyre and Muni Sreenivas Pydi.*

This is a codebase to replicate the results of the blogpost: [blogpost]. It allows to visualize small attention-only Transformers with embedding dimension 3. 
* models.py contains the architecture of the Transformer (which is standard),
* train.py is used to train the Transformer normally of using head-boosting (training heads individually),
* interp.py contains the main fonctions to visualize the Transformer,

Use the notebook to see how to use the codebase, and replicate the results.