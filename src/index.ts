import { NeuralNetwork } from "./NeuralNetwork";
import { E_GameState, TicTacToeGame } from "./TicTacToeGame";

const nn = new NeuralNetwork([2,4,4,4]);
//nn.init();
//nn.train(.2);

const g = new TicTacToeGame();
const w = new Worker("train.js");
w.onmessage = (message) =>
{
    const data = JSON.parse(message.data);
    g.neuralNetwork.setWeights(data.weights, data.biases);
};


//g.train(1000);
g.board.reset();


const playMove = () =>
{
    if (g.board.gameState === E_GameState.RUNNING)
    {
        g.board.makeMove(document.getElementsByClassName('input')[0].value, 0);

        if (g.board.gameState === E_GameState.RUNNING)
        {
            const bestMove = g.findBestMove();
            g.board.makeMove(bestMove.pos, 0);
        }
    }

    document.getElementById("result").innerText = g.board.toString();
};

const resetGame = () =>
{
    g.board.reset();
    document.getElementById("result").innerText = g.board.toString();
};


const basicTest = () =>
{
    const inputs = document.getElementsByClassName('input');
    const inArr = [];
    for (let i = 0; i < inputs.length; i++)
    {
        const input = inputs[i];
        inArr.push(input.value);
    }

    nn.forwardPass(inArr);
    nn.backPass(nn.generateOuts(inArr));
    // let s = nn.test(inArr, nn.generateOuts(inArr));
    let s = "";

    nn.getOutputs().forEach( (v,i) => s += ("OUT " + i + " > " + v.toFixed(4) + "<br/>"));

    s += "<br/>Error : " + nn.calcError(nn.generateOuts(inArr)).toFixed(4);
    document.getElementById("result").innerHTML = s;
};


document.getElementById('submitBut').onclick = playMove;
document.getElementById('resetBut').onclick = resetGame;


