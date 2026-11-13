# Neural Network in Haskell

A small neural network library written from scratch in **pure functional Haskell**, with no machine learning libraries. It is trained with backpropagation to learn the **XOR** function.

Group project for the course *Programação Funcional* (BSc in Computer Engineering, Faculdade de Ciências da Universidade de Lisboa, 2025/2026).

---

## What it does

XOR is the classic example of a problem that is **not linearly separable**: no single straight line separates the true cases from the false ones. A network without a hidden layer cannot learn it, so this project uses a **2 → 4 → 1** architecture (2 inputs, a hidden layer of 4 neurons and 1 output).

| x1 | x2 | x1 XOR x2 |
|:--:|:--:|:---------:|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

```
 Input        Hidden layer       Output

  x1 ──┬──────▶ h1 ──┐
       ├──────▶ h2 ──┤
       ├──────▶ h3 ──┼──────▶ ŷ
  x2 ──┴──────▶ h4 ──┘
```

Each neuron computes `σ(W·x + b)`, where `σ` is the sigmoid function.

## How it works

1. **Build the network**: `buildNetwork 2 [4,1] weights` creates the layers from a list of initial weights. When training, this list is an infinite stream of random values in `[-1, 1]`.
2. **Forward pass**: `forwardPass` propagates the input layer by layer and returns the activations of every layer, which backpropagation needs.
3. **Backpropagation**: `backPropagation` computes the error signal (delta) of each layer from the output backwards, then updates every weight and bias with gradient descent.
4. **Training**: `training` repeats this for 100 000 epochs over the four XOR examples, with a learning rate of 0.1.
5. **Saving**: the trained network is written to a file with `show` and loaded back with `read`. The `Layer` type derives `Show` and `Read`, so no custom parser is needed.

The code is written in a functional style: pattern matching, guards, recursion and higher-order functions (`map`, `foldl`, `zipWith`, `zipWith3`), with no mutable state. Only `main`, `trainMode`, `predictMode` and `testMode` run in `IO`; everything else is pure.

## Project structure

| File | Contents |
|---|---|
| `projecto.hs` | Entry point: command-line interface (`--train`, `--predict`, `--test`) |
| `NeuralNetwork.hs` | Vector and matrix operations, sigmoid, the `Layer` and `Network` types, error functions (MSE), forward pass and backpropagation |
| `Training.hs` | Training loop |
| `Test.hs` | QuickCheck properties |

## Tests

The properties are checked with **QuickCheck** on randomly generated inputs:

- **Sigmoid**: the output is always in the interval `[0, 1]`
- **MSE**: the mean squared error is never negative
- **Backpropagation**: after one training step, the network keeps exactly the same shape (number of layers, neurons per layer and weights per neuron)

## How to run

Requirements: [GHC](https://www.haskell.org/ghcup/) and the `QuickCheck` and `random` packages.

```bash
cabal install --lib QuickCheck random
ghc projecto.hs -o projecto
```

**Train** the network. The weights are saved to the file, and the prediction for each XOR input and the final error are printed:

```bash
./projecto --train pesos.txt
```

**Predict** with the saved weights:

```bash
./projecto --predict pesos.txt
Input 1: 1
Input 2: 0
Output: [1]
```

**Run the tests:**

```bash
./projecto --test
```

## Authors

- Cristiano Zhou
- Shangchen Lin
