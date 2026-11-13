-- Autores:
--          Cristiano Zhou 65581
--          Shangchen Lin  65645

-- Parte de entrega 1 
module NeuralNetwork where 

-- Exemplo: chunksOf 2 [1, 2, 3, 4, 5, 6] == [[1, 2], [3, 4], [5, 6]]
chunksOf :: Int -> [a] -> [[a]]
chunksOf n _
    | n < 0 = error "não pode ser valor negativos!!!"
chunksOf 0 _ = []
chunksOf _ [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

-- Exemplo: sigmoid 100.0 == 1.0
--          sigmoid (-100.0) == 3.7200759760208356e-44 == 0.0
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x)) 

-- Exemplo: sigmoid' 0.01 == 9.9e-3 == 0.0099
--          sigmoid' 0.99 == 9.90000000000001e-3 == 0.0099
sigmoid' :: Double -> Double
sigmoid' y = y * (1 - y)

-- Exemplo: transpose [[1, 3, 4, 0], [2, -1, 0, 1]] == [[1,2], [3, -1], [4, 0], [0, 1]]
transpose :: [[a]] -> [[a]]
transpose [] = []
transpose xs 
    | any null xs = [] -- é para não acontecer erro 
    | otherwise = map head xs : transpose (map tail xs)

-- Exemplo: multMatrix [[2, 0], [0, 0.5]] [10, 20] == [20.0, 10.0]
multMatrix :: [[Double]] -> [Double] -> [Double]
multMatrix [] _ = error "Não pode ser uma matriz vazia!!!"
multMatrix _ [] = error "Não pode ser um vetor vazio!!!"
multMatrix m x 
    | any (\row -> length row /= length x) m = error "Dimensões incompatíveis!"
    | otherwise = map (sum . zipWith (*) x) m 

-- Exemplo: somaVectorial [2.0, 5.0] [12.0, 19.0] == [14.0, 24.0]
somaVectorial :: [Double] -> [Double] -> [Double]
somaVectorial xs ys 
    | length xs /= length ys = error "Vetores com tamanhos diferentes!!!"
    | otherwise = zipWith (+) xs ys 

-- Parte de entraga 2

-- tipos pedidos
data Layer = Layer [[Double]] [Double] deriving (Show, Read)
type Network = [Layer]

-- Exemplo: buildNetwork 1 [1] [0.5, 0.3] == [Layer [[0.5]] [0.3]]
buildNetwork :: Int -> [Int] -> [Double] -> Network
buildNetwork _ [] _ = [] 
buildNetwork inputSize (x:xs) ys = layer : buildNetwork x xs resto
    where
        peso = take (x * inputSize) ys
        bias = take x (drop (x * inputSize) ys)
        resto = drop (x * inputSize + x) ys
        layer = Layer (chunksOf inputSize peso) bias

-- Exemplo: outputError [0.7] [0.5] == [0.2]
outputError :: [Double] -> [Double] -> [Double]
outputError = zipWith (-)

-- Exemplo: mse [1.0] [0.0] == 1.0
mse :: [Double] -> [Double] -> Double
mse previsao alvo = sum errosQuadratico / fromIntegral (length erros)
    where
        erros = outputError previsao alvo 
        errosQuadratico = map (^2) erros

-- Exemplo: msePredictions [[1.0],[0.0]] [[0.0],[0.0]] == 0.5
msePredictions :: [[Double]] -> [[Double]] -> Double
msePredictions previsoes alvos = sum mseIndividuais / fromIntegral (length mseIndividuais)
    where
        mseIndividuais = zipWith mse previsoes alvos 

-- Exemplo: length (forwardPass [0,1] (buildNetwork 2 [4,1] (repeat 0.1))) == 3
forwardPass :: [Double] -> Network -> [[Double]]
forwardPass xs [] = [xs]
forwardPass xs (Layer ws bs : resto) = xs : forwardPass layerOutput resto 
    where 
        mult= multMatrix ws xs
        soma = somaVectorial mult bs 
        layerOutput = map sigmoid soma

-- Exemplo: length (backPropagation 0.1 [0,1] [1.0] (buildNetwork 2 [4,1] (repeat 0.1))) == 2
backPropagation :: Double -> [Double] -> [Double] -> Network -> Network
backPropagation taxaAprendizagem input esperado nw = zipWith3 (updateLayer taxaAprendizagem) nw deltas atAnterior
    where 
        -- passo 1 
        ativacoes = forwardPass input nw 
        atAnterior = init ativacoes 
        outputAt = last ativacoes 

        -- passo 2 
        outputDelta = outputError outputAt esperado     

        -- passo 3
        deltas = calcDeltas (reverse nw) outputDelta (reverse atAnterior)
            where
                calcDeltas :: Network -> [Double] -> [[Double]] -> [[Double]]
                calcDeltas [] _ _ = []
                calcDeltas (Layer ws bs : restoPesos) delta (at:restoAt) =
                        calcDeltas restoPesos deltaCalc restoAt ++ [delta]
                        where
                            transPesos = multMatrix (transpose ws) delta 
                            deltaCalc = zipWith (\w a -> w * sigmoid' a) transPesos at
        -- passo 4 
        updateLayer :: Double -> Layer -> [Double] -> [Double] -> Layer
        updateLayer taxaAprendizagem (Layer ws bs) deltas atAnterior = Layer newWs newBs
            where
                newWs = zipWith (\row d -> zipWith (\w a -> w - taxaAprendizagem * d * a) row atAnterior) ws deltas 
                newBs = zipWith (\b d -> b - taxaAprendizagem * d) bs deltas