# PaliGemma: Multimodal Vision-Language Model

## Architecture Overview


### visual explanation of the _merge_input_ids_with_image_features function:
Input Sequence Structure:
[IMAGE][IMAGE][IMAGE][BOS][TEXT][TEXT][TEXT][EOS]
   ^     ^     ^      ^    ^    ^     ^     ^
   |     |     |      |    |    |     |     |
   +-Image Tokens-+   +----Text Tokens-----+

1. CREATING MASKS:
   Input IDs:  [500, 500, 500,  1,  70,  71,  75,   2]
               └─image tokens─┘ └────text tokens────┘
   
   Text Mask:  [ 0,  0,  0,   1,   1,   1,   1,   1]
   Image Mask: [ 1,  1,  1,   0,   0,   0,   0,   0]
   Pad Mask:   [ 0,  0,  0,   0,   0,   0,   0,   0]

2. MERGING PROCESS:
   ┌─────────────────────────────────────────────┐
   │ Image Features   Text Embeddings            │
   │ [■][■][■]       [□][□][□][□][□]           │
   └─────────────────────────────────────────────┘
              ↓ (merge)
   ┌─────────────────────────────────────────────┐
   │ Final Embedding                             │
   │ [■][■][■][□][□][□][□][□]                  │
   └─────────────────────────────────────────────┘
   Where: ■ = Image embedding
          □ = Text embedding

3. ATTENTION MASK (for generation):
   ┌─────────────┐
   │ Can attend  │
   │ ↓           │
   │ [1][1][1]...│ Query token
   └─────────────┘

4. POSITION IDS:
   [0][1][2][3][4][5][6][7]
   Sequential positions for rotary embeddings
