import {E_GameState, E_LineState, E_Player} from "./TicTacToeGame";

export class TicTacToeBoard
{


    static LINES =
    [
        [0,1,2],
        [3,4,5],
        [6,7,8],
        [0,3,6],
        [1,4,7],
        [2,5,8],
        [0,4,8],
        [2,4,6]
    ];

    public state:number[];
    public lineState:number[];
    public lineCounter:number[];
    public numOpenLines:number;
    public gameState:E_GameState;
    public playerToMove:E_Player;
    public movesList:number[];
    public scores:number[];

    private readonly availableMoves:number[];
    private peekPos:number;

    constructor()
    {
        this.state = [];
        this.availableMoves = [];
        this.movesList = [];
        this.scores = [];
    }


    reset()
    {
        for (let boardIdx=0; boardIdx<9; boardIdx++)
        {
            this.state[boardIdx] = 0;
        }

        for (let lineIdx=0; lineIdx<8; lineIdx++)
        {
            this.lineState[lineIdx] = E_LineState.OPEN;
            this.lineCounter[lineIdx] = 3;
        }

        this.numOpenLines = this.lineState.length;
        this.gameState = E_GameState.RUNNING;
        this.playerToMove = E_Player.PLAYER_X;
        this.movesList.length = 0;
        this.scores.length = 0;
    }


    makeMove(posIndex)
    {
        if (this.state[posIndex] !== 0 || this.gameState !== E_GameState.RUNNING)
        {
            debugger;
        }

        this.state[posIndex] = this.playerToMove;

        const numLines = TicTacToeBoard.LINES.length;
        for (let lineIdx=0; lineIdx < numLines; lineIdx++)
        {
            const line = TicTacToeBoard.LINES[lineIdx];
            if (line[0] === line[1] && line[1] === line[2] && line[0] !== 0)
            {
                this.gameState = line[0];
            }
        }

        this.movesList.push(posIndex);
        this.playerToMove *= -1;
    }


    undoMove(movePos:number):void
    {
        this.gameState = E_GameState.RUNNING;
        this.state[movePos] = 0;
    }


    getAvailableMoves()
    {
        this.availableMoves.length = 0;
        for(let posIdx=0; posIdx < this.state.length; posIdx++)
        {
            if (this.state[posIdx] === E_Player.NONE)
            {
                this.availableMoves.push(posIdx);
            }
        }
        return this.availableMoves;
    }


    toString():string
    {
        const MAP =
        {
            [E_Player.NONE] : "_",
            [E_Player.PLAYER_X] : "X",
            [E_Player.PLAYER_O] : "O",
        };
        let s = "";
        for (let boardIdx = 0; boardIdx < this.state.length; boardIdx++)
        {
            s += MAP[this.state[boardIdx]] + " ";
            if ((boardIdx+1) % 3 === 0)
            {
                s+="\n";
            }
        }
        return s;
    }


    markMove(posIdx:number):void
    {
        this.state[posIdx] = this.playerToMove;
        this.peekPos = posIdx;
    }


    unmarkMove():void
    {
        this.state[this.peekPos] = 0;
    }


}