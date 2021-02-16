import {NeuralNetwork} from "./NeuralNetwork";
import {TicTacToeBoard} from "./TicTacToeBoard";

export class TicTacToeGame
{

    public board:TicTacToeBoard;
    public neuralNetwork:NeuralNetwork;


    constructor()
    {
        this.board = new TicTacToeBoard();
        this.neuralNetwork = new NeuralNetwork([9,100,100,100,1]);
        this.neuralNetwork.init();
    }

    train(numEpochs:number)
    {
        for (let epochIdx=0; epochIdx < numEpochs; epochIdx++)
        {
            const batchSize = 10;
            let batchError = 0;

            for (let batchIdx=0; batchIdx < batchSize; batchIdx++)
            {
                this.board.reset();
                while(this.board.gameState === E_GameState.RUNNING)
                {
                    const bestMove = this.findBestMove();
                    this.board.makeMove(bestMove.pos, bestMove.score);
                }

                let currScore = (this.board.gameState + 1) * .5;
                let scoreStep = (.5 - currScore) / this.board.movesList.length;

                while(this.board.movesList.length > 0)
                {
                    this.neuralNetwork.forwardPass(this.board.state);
                    this.neuralNetwork.backPass([currScore]);

                    batchError += this.neuralNetwork.calcError([currScore]);

                    currScore += scoreStep;

                    const move = this.board.movesList.pop();
                    this.board.state[move] = 0;
                }
            }

            this.neuralNetwork.applyTrainVector(.2, batchSize, 1);
            this.neuralNetwork.resetTrainVector(batchSize);

            if (epochIdx % 10 === 0)
            {
                console.log("epoch = " + epochIdx + ", batch error = " + batchError/batchSize);
            }
        }
    }


    findBestMove()
    {
        const positions = this.board.getAvailableMoves();
        let selectedMovePos;
        let maxScore = Number.NEGATIVE_INFINITY;
        for (let posIdx=0; posIdx < positions.length; posIdx++)
        {
            this.board.markMove(positions[posIdx]);
            this.neuralNetwork.forwardPass(this.board.state);
            const nnOut = this.neuralNetwork.getOutputs()[0];
            const score = (.5 - nnOut) * -this.board.playerToMove;
            if (score > maxScore)
            {
                maxScore = score;
                selectedMovePos = positions[posIdx];
            }
            this.board.unmarkMove();
        }
        return {pos:selectedMovePos, score:maxScore};
    }

}


export enum E_Player
{
    PLAYER_X = 1,
    NONE = 0,
    PLAYER_O = -1
}


export enum E_GameState
{
    DRAW = 0,
    WIN_X = 1,
    WIN_O = -1,
    RUNNING = 2
}

export enum E_LineState
{
    OPEN = 0,
    PLAYER_X = 1,
    PLAYER_O = -1,
    BLOCKED = 2,
}
