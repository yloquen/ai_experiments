import {E_GameState, E_LineState, E_Player} from "./TicTacToeGame";

export class TicTacToeBoard
{


    static POSITION_TO_LINE =
    [
        [0,3,6],
        [1,3],
        [2,3,7],
        [0,4],
        [1,4,6,7],
        [2,4],
        [0,5,7],
        [1,5],
        [2,5,6]
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
        this.lineState = [];
        this.lineCounter = [];
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


    makeMove(posIndex, score)
    {
        if (this.state[posIndex] !== 0 || this.gameState !== E_GameState.RUNNING)
        {
            debugger;
        }

        this.scores.push(score);
        this.state[posIndex] = this.playerToMove;
        const lines = TicTacToeBoard.POSITION_TO_LINE[posIndex];

        for (let i=0; i < lines.length; i++)
        {
            const lineIdx = lines[i];
            const lineState = this.lineState[lineIdx];
            if (lineState === E_LineState.OPEN || lineState === this.playerToMove)
            {
                this.lineState[lineIdx] = this.playerToMove;
                this.lineCounter[lineIdx]--;
                if (this.lineCounter[lineIdx] === 0)
                {
                    this.gameState = this.playerToMove;
                }
            }
            else if (lineState !== this.playerToMove)
            {
                if (this.lineState[lineIdx] !== E_LineState.BLOCKED)
                {
                    this.lineState[lineIdx] = E_LineState.BLOCKED;
                    this.numOpenLines--;
                    if (this.numOpenLines === 0)
                    {
                        this.gameState = E_GameState.DRAW;
                    }
                }
            }
        }
        this.movesList.push(posIndex);
        this.playerToMove *= -1;
    }


    undoMove(movePos)
    {

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