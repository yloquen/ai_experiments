import {TicTacToeGame} from "./TicTacToeGame";


const g = new TicTacToeGame();
g.train(1);

postMessage(JSON.stringify({weights:g.neuralNetwork.weights, biases:g.neuralNetwork.biases}));


