import {NeuralNetwork} from "./NeuralNetwork";
import {TicTacToeBoard} from "./TicTacToeBoard";

export class TicTacToeGame
{

    public board:TicTacToeBoard;
    public neuralNetwork:NeuralNetwork;


    constructor()
    {
        this.board = new TicTacToeBoard();
        this.neuralNetwork = new NeuralNetwork([9,10,10,10,1]);
        this.neuralNetwork.init();
    }


    monteCarloSearch()
    {
        this.neuralNetwork.forwardPass(this.board.state);
        const score = this.neuralNetwork.getOutputs()[0];

        const moves = this.board.getAvailableMoves();

        for (let moveIdx=0; moveIdx < moves.length; moveIdx++)
        {
            const movePos = moves[moveIdx];
            this.board.makeMove(movePos);

            this.board.undoMove(movePos);
        }
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
                this.monteCarloSearch();
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
