-- Autores:
--          Cristiano Zhou 65581
--          Shangchen Lin  65645

-- Parte de entrega 3
module Main where
import NeuralNetwork
import Training
import Test
import System.Environment (getArgs)
import Test.QuickCheck
import System.Random (randomRs, newStdGen) 
import System.IO (hSetBuffering, stdout, BufferMode(..))
import Text.Printf (printf)

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["--train", fp] -> trainMode fp
        ["--predict", fp] -> predictMode fp
        ["--test"] -> testMode
        _ -> putStrLn "Uso: --train <ficheiro> | --predict <ficheiro> | --test"

-- inputs possiveis 
xorInputs :: [[Double]]
xorInputs = [[0,0], [0,1], [1,0], [1,1]]

-- outputs esperados 
xorOutputs :: [[Double]]
xorOutputs = [[0], [1], [1] , [0]]

-- mode para treinar 
trainMode :: FilePath -> IO ()
trainMode fp = do
    gen <- newStdGen
    let pesos = randomRs (-1.0, 1.0) gen
        nwInicial = buildNetwork 2 [4,1] pesos
        nwTrained = training 100000 0.1 (xorInputs, xorOutputs) nwInicial
        previsoes = map (\x -> last (forwardPass x nwTrained)) xorInputs
        mseFinal = msePredictions previsoes xorOutputs
        pares = zip previsoes xorOutputs
    writeFile fp $ show nwTrained
    putStrLn $ "Pesos guardados em " ++ fp
    putStrLn $ "[" ++ init (concatMap (\(p, y) -> printf "([%.5f],[%.5f])," (head p) (head y)) pares) ++ "]"
    printf "MSE: %.2e\n" mseFinal

-- mode de input e output 
predictMode :: FilePath -> IO ()
predictMode fp = do
    hSetBuffering stdout NoBuffering
    conteudo <- readFile fp
    putStr "Input 1: "
    x1 <- readLn :: IO Double
    putStr "Input 2: "
    x2 <- readLn :: IO Double
    let nw = read conteudo :: Network
        resultado = last (forwardPass [x1, x2] nw)
    putStrLn ("Output: " ++ show (map round resultado))

-- mode para quickcheck
testMode :: IO ()
testMode = do
    putStrLn "Teste de sigmoid:"
    quickCheck propSigmoid
    putStrLn "Teste de Mse:"
    quickCheck propMse
    putStrLn "Teste de BackPropagation:"
    quickCheck propBack
