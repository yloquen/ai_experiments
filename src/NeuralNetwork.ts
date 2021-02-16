export class NeuralNetwork
{
    public weights:number[][][];

    private readonly layers:number[];
    private readonly activations:number[][];
    private readonly activationDerivatives:number[][];

    private readonly weightVector:number[][][];
    private biases:number[][];
    private readonly biasVector:number[][];
    private readonly dEdZ:number[][];


    constructor(layers:number[])
    {
        this.layers = layers;
        this.activations = [];
        this.activationDerivatives = [];
        this.weights = [];
        this.weightVector = [];
        this.biases = [];
        this.biasVector = [];
        this.dEdZ = [];
    }


    init():void
    {
        for (let layerIdx=0; layerIdx < this.layers.length; layerIdx++)
        {
            const numNeurons = this.layers[layerIdx];

            this.activations[layerIdx] = [];
            this.activationDerivatives[layerIdx] = [];
            this.weights[layerIdx] = [];
            this.weightVector[layerIdx] = [];
            this.biases[layerIdx] = [];
            this.biasVector[layerIdx] = [];
            this.dEdZ[layerIdx] = [];

            for (let neuronIdx=0; neuronIdx < numNeurons; neuronIdx++)
            {
                this.biases[layerIdx][neuronIdx] = this.getInitVal();
                this.biasVector[layerIdx][neuronIdx] = 0;
                this.activations[layerIdx][neuronIdx] = 0;
                this.activationDerivatives[layerIdx][neuronIdx] = 0;
                if (layerIdx < this.layers.length-1)
                {
                    const nextNumNeurons = this.layers[layerIdx+1];
                    const weights = [];
                    this.weights[layerIdx].push(weights);
                    const weightVector = [];
                    this.weightVector[layerIdx].push(weightVector);
                    for (let nextNeuronIdx=0; nextNeuronIdx < nextNumNeurons; nextNeuronIdx++)
                    {
                        weights.push(this.getInitVal());
                        weightVector.push(0);
                    }
                }
            }
        }
    }


    getInitVal():number
    {
        return -1 + Math.random() * 2;
    }


    forwardPass(inputs)
    {
        this.activations[0] = inputs;

        for (let layerIdx = 0; layerIdx < this.activations.length-1; layerIdx++)
        {
            for (let inIdx = 0; inIdx < this.activations[layerIdx+1].length; inIdx++)
            {
                let z = 0;
                for (let outIdx = 0; outIdx < this.activations[layerIdx].length; outIdx++)
                {
                    z += this.activations[layerIdx][outIdx] * this.weights[layerIdx][outIdx][inIdx];
                }
                z += this.biases[layerIdx+1][inIdx];
                const activationVal = this.activationFunc(z);
                this.activations[layerIdx+1][inIdx] = activationVal;
                this.activationDerivatives[layerIdx+1][inIdx] = this.activationDerivative(activationVal);
            }
        }
    }


    activationFunc(x):number
    {
        return 1 / (1 + Math.exp(-x));
    }


    activationDerivative(x:number):number
    {
        return x * (1 - x);
    }


    backPass(targetOutputs)
    {
        let layerIdx = this.layers.length-1;

        for (let neuronIdx=0; neuronIdx < this.activations[layerIdx].length; neuronIdx++)
        {
            const actVal = this.activations[layerIdx][neuronIdx];
            this.dEdZ[layerIdx][neuronIdx] = (targetOutputs[neuronIdx] - actVal) * this.activationDerivatives[layerIdx][neuronIdx];
        }

        while(layerIdx > 0)
        {
            const numNeurons = this.layers[layerIdx];
            const numPrevNeurons = this.layers[layerIdx-1];
            const sigmoidDerivativeLayer = this.activationDerivatives[layerIdx-1];
            const dEdZLayer = this.dEdZ[layerIdx];
            const activationsLayer = this.activations[layerIdx-1];

            for (let prevNeuronIdx=0; prevNeuronIdx < numPrevNeurons; prevNeuronIdx++)
            {
                let sum=0;
                const weightVectorLayer = this.weightVector[layerIdx-1][prevNeuronIdx];
                for (let neuronIdx=0; neuronIdx < numNeurons; neuronIdx++)
                {
                    const d = dEdZLayer[neuronIdx];
                    const w = this.weights[layerIdx-1][prevNeuronIdx][neuronIdx];
                    weightVectorLayer[neuronIdx] += d * activationsLayer[prevNeuronIdx];
                    sum += d*w;
                }
                this.dEdZ[layerIdx-1][prevNeuronIdx] = sum  * sigmoidDerivativeLayer[prevNeuronIdx];
            }

            const biasVectorLayer = this.biasVector[layerIdx];
            for (let neuronIdx=0; neuronIdx < numNeurons; neuronIdx++)
            {
                biasVectorLayer[neuronIdx] += dEdZLayer[neuronIdx];
            }

            layerIdx--;
        }
    }


    test(inputs, outputs)
    {
        const delta = .0001;

        let s = "";
        for (let layerIdx=0; layerIdx < this.weights.length; layerIdx++)
        {
            s += "<br/>";
            for (let inIdx=0; inIdx < this.weights[layerIdx].length; inIdx++)
            {
                for (let weightIdx=0; weightIdx < this.weights[layerIdx][inIdx].length; weightIdx++)
                {
                    s += ("W(" + layerIdx + ")" + inIdx + "," + weightIdx + ". . . . .");
                    const weightsCopy = JSON.parse(JSON.stringify(this.weights));
                    this.forwardPass(inputs);
                    const error = this.calcError(outputs);
                    this.weights[layerIdx][inIdx][weightIdx] += delta;
                    this.forwardPass(inputs);
                    this.backPass(outputs);
                    const deltaError = this.calcError(outputs) - error;
                    const derivative = deltaError/delta;
                    s += (derivative.toFixed(8) + " : " + this.weightVector[layerIdx][inIdx][weightIdx].toFixed(8) + "<br/>");
                    this.weights = weightsCopy;
                }

                s += ("B(" + layerIdx + ")" + inIdx + ". . . . .");
                const biasesCopy = JSON.parse(JSON.stringify(this.biases));
                this.forwardPass(inputs);
                const error = this.calcError(outputs);
                this.biases[layerIdx][inIdx] += delta;
                this.forwardPass(inputs);
                const deltaError = this.calcError(outputs) - error;
                const derivative = deltaError/delta;
                s += (derivative.toFixed(8) + " : " + this.biasVector[layerIdx][inIdx].toFixed(8) + "<br/>");
                this.biases = biasesCopy;
            }

        }

        return s;
    }


    calcError(targetOutputs)
    {
        const activationLayer = this.activations[this.activations.length-1];
        let totalError = 0;
        for (let outIdx=0; outIdx < targetOutputs.length; outIdx++)
        {
            const diff = targetOutputs[outIdx] - activationLayer[outIdx];
            totalError +=  .5 * diff * diff;
        }
        return totalError;
    }


    applyTrainVector(rate, batchSize, sign)
    {
        const batchSizeReciprocal = 1 / batchSize;
        for (let layerIdx=0; layerIdx < this.weights.length; layerIdx++)
        {
            for (let inIdx = 0; inIdx < this.weights[layerIdx].length; inIdx++)
            {
                for (let weightIdx = 0; weightIdx < this.weights[layerIdx][inIdx].length; weightIdx++)
                {
                    this.weights[layerIdx][inIdx][weightIdx] += sign * this.weightVector[layerIdx][inIdx][weightIdx] * batchSizeReciprocal * rate;
                }
                this.biases[layerIdx][inIdx] += this.biasVector[layerIdx][inIdx] * batchSizeReciprocal * rate;
            }
        }
    }


    resetTrainVector(batchSize)
    {
        const batchSizeReciprocal = 1 / batchSize;
        for (let layerIdx=0; layerIdx < this.weights.length; layerIdx++)
        {
            for (let inIdx = 0; inIdx < this.weights[layerIdx].length; inIdx++)
            {
                for (let weightIdx = 0; weightIdx < this.weights[layerIdx][inIdx].length; weightIdx++)
                {
                    this.weightVector[layerIdx][inIdx][weightIdx] = 0;
                }
                this.biasVector[layerIdx][inIdx] = 0;
            }
        }
    }

    setWeights(weights, biases)
    {
        for (let layerIdx=0; layerIdx < this.weights.length; layerIdx++)
        {
            for (let inIdx = 0; inIdx < this.weights[layerIdx].length; inIdx++)
            {
                for (let weightIdx = 0; weightIdx < this.weights[layerIdx][inIdx].length; weightIdx++)
                {
                    this.weights[layerIdx][inIdx][weightIdx] = weights[layerIdx][inIdx][weightIdx];
                }
                this.biases[layerIdx][inIdx] = biases[layerIdx][inIdx];
            }
        }
    }



    generateIns()
    {
        return [Number(Math.random() > .5), Number(Math.random() > .5)];
    }


    generateOuts(ins)
    {
        const outs = [0,0,0,0];
        outs[ins[0] | ins[1] << 1] = 1;
        return outs;
    }


    getOutputs()
    {
        return this.activations[this.layers.length-1];
    }



}