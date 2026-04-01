# Clavier

<p align="left">
    <img src="assets/bach_head.png" alt="orrp" height="100px" width="100px" />
</p>

Clavier is a 12M parameter language model that generates sheet music and MIDI files for new compositions in the style of classical master composers like Bach, Beethoven, Chopin, etc. 

Example model output:

<p align="left">
    <a href="generated/composition_01.pdf">
        <img src="assets/composition_01.png" alt="orrp" height="350px" width="300px"  />
    </a>
</p>

[ABC Notation](generated/composition_01.abc)

[MIDI file](generated/composition_01.mid)

## Architecture

The input data for Clavier is the [ABC notation format](https://en.wikipedia.org/wiki/ABC_notation), which is a file format specifically made for computers to represent sheet music. The ABC notation files were scraped from the web or converted from MIDI files that were also scraped from the web. Before pre-processing the data, a script is run to [transpose](https://en.wikipedia.org/wiki/Transposition_(music)) each piece into the 11 other musical keys. This data augmentation step increases the data set size and helps to prevent overfitting. This is a similar technique used for computer vision, such as CNNs, where images are rotated randomly. 

A custom Hugging Face tokenizer was built from the ABC training data after a pre-processing step that has a number of gatekeeper checks, such as limiting compositions with more than four voices to minimize context window bloat and complexity. The tokenizer is a custom [BPE tokenizer](https://en.wikipedia.org/wiki/Byte-pair_encoding) with a vocabulary of 2000 tokens. The tokenizer uses a number of special tokens, e.g., a token that indicates the composer of the piece so that the model can learn the styles of different composers. Pre-tokenization is also utilized so that the ABC data is split into more useful chunks before tokenization to improve training performance. The pre-processing step also does a 90/10 validation split to improve evaluation of the model with unseen data. 

The Clavier model itself is an autoregressive decoder-only transformer-based language model with 12 million parameters. It uses positional embeddings, causal multi-head self-attention, Flash Attention, and GELU activations.

The model was trained using Google Colab with a T4 GPU. Training utilizes step-based validation every 3,000 steps, at which point the model does a random sampling of predictions across a validation/unseen dataset to get an average loss.

## Generation

The generation script will utilize the Clavier model to predict the next sequence of notes and timing based on the style of the composer in question. The generation script also performs some clean up to ensure that the ABC notation is properly formatted. Command line tools are then used to convert the ABC output to both sheet music (PDF) and a MIDI file for playback. 

## Contributions

Please open a GitHub issue or PR to contribute.
