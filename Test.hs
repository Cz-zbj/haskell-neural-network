-- Autores:
--          Cristiano Zhou 65581
--          Shangchen Lin  65645

-- quickcheck
module Test where 
import NeuralNetwork
import Test.QuickCheck 
import Data.List (repeat)

instance Arbitrary Layer where
    arbitrary = do
        numNeurons <- choose (1, 5)
        numInputs <- choose (1, 5)
        ws <- vectorOf numNeurons (vectorOf numInputs arbitrary)
        bs <- vectorOf numNeurons arbitrary
        return (Layer ws bs)

-- primeira propriedade 
-- testar se o sigmoid fique com intervalo de 0 e 1
propSigmoid :: Double -> Bool
propSigmoid x = sigmoid x >= 0 && sigmoid x <= 1 -- existe o sinal de igual para passar os teste por a causa de arredodamentos 

-- segunda propriedade 
-- testar se o valor de mse é sempre positivo 
propMse :: [Double] -> [Double] -> Bool
propMse xs ys
    | null xs || null ys = True -- guarda pq n faz sentido ao mse 
    | length xs /= length ys = True -- mesma coisa 
    | otherwise = mse xs ys >= 0

-- funcao auxiliar
genNetwork :: Gen Network
genNetwork = do
    return $ buildNetwork 2 [4, 1] (repeat 0.1)

-- terceira propriedade 
propBack :: Property
propBack = forAll genNetwork $ \nw ->
    let 
        input = [0.0, 0.0]
        esperado = [1.0]
        nwNovo = backPropagation 0.1 input esperado nw
        camada :: Layer -> (Int, [Int])
        camada (Layer ws _) = (length ws, map length ws)
    in map camada nw == map camada nwNovo