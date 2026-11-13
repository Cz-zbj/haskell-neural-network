-- Autores:
--          Cristiano Zhou 65581
--          Shangchen Lin  65645

-- modulo para treinar
module Training where 
import NeuralNetwork

-- Exemplo: 10 0.1 ([[0,0]], [[0]]) (buildNetwork 2 [4,1] (repeat 0.1))
training :: Int -> Double -> ([[Double]], [[Double]]) -> Network -> Network
training 0 _ _ nw = nw
training n taxa (xs, ys) nw = training (n -1) taxa (xs, ys) nwNew
    where 
        testes = zip xs ys 
        nwNew = foldl (\net (xss, yss) -> backPropagation taxa xss yss net) nw testes 