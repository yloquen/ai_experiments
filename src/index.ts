const LAYERS = [2,3,4,3];

const activations = [];
const sigmoidDerivatives = [];
let weights = [];
const weightVector = [];
const biasVector = [];
const biases = [];
const dEdZ = [];
let cnt = 0;



function rnd(seed)
{
    cnt++;
    return Math.sin((cnt + 28983) * Math.pow(5, seed * 3));
}



function init()
{
    for (let layerIdx=0; layerIdx < LAYERS.length; layerIdx++)
    {
        const numNeurons = LAYERS[layerIdx];

        activations[layerIdx] = [];
        sigmoidDerivatives[layerIdx] = [];
        weights[layerIdx] = [];
        weightVector[layerIdx] = [];
        biases[layerIdx] = [];
        dEdZ[layerIdx] = [];

        for (let neuronIdx=0; neuronIdx < numNeurons; neuronIdx++)
        {
            biases[layerIdx][neuronIdx] = 0;
            activations[layerIdx][neuronIdx] = 0;
            sigmoidDerivatives[layerIdx][neuronIdx] = 0;
            if (layerIdx < LAYERS.length-1)
            {
                const nextNumNeurons = LAYERS[layerIdx+1];
                const w = [];
                weights[layerIdx].push(w);
                const wv = [];
                weightVector[layerIdx].push(wv);
                for (let nextNeuronIdx=0; nextNeuronIdx < nextNumNeurons; nextNeuronIdx++)
                {
                    w.push(rnd(1));
                    wv.push(0);
                }
            }
        }
    }
}


function forwardPass(inputs)
{
    activations[0] = inputs;

    for (let layerIdx = 0; layerIdx < activations.length-1; layerIdx++)
    {
        let zSum=0;
        for (let inIdx = 0; inIdx < activations[layerIdx+1].length; inIdx++)
        {
            let z = 0;
            for (let outIdx = 0; outIdx < activations[layerIdx].length; outIdx++)
            {
                if (layerIdx === 1 && inIdx === 1 && outIdx ===0)
                {
                    // weights[layerIdx][outIdx][inIdx] += .0001;
                }
                z += activations[layerIdx][outIdx] * weights[layerIdx][outIdx][inIdx];
            }
            z += biases[layerIdx+1][inIdx];
            zSum += z;
            let sigmoidVal = sigmoid(z);

            activations[layerIdx+1][inIdx] = sigmoidVal;
            sigmoidDerivatives[layerIdx+1][inIdx] = sigmoidVal * (1 - sigmoidVal);
        }
    }
}


function sigmoid(x)
{
    return 1 / (1 + Math.exp(-x));
}


function backPass(targetOutputs)
{
    let layerIdx = LAYERS.length-1;

    for (let neuronIdx=0; neuronIdx < activations[layerIdx].length; neuronIdx++)
    {
        const actVal = activations[layerIdx][neuronIdx];
        dEdZ[layerIdx].push((targetOutputs[neuronIdx] - actVal) * actVal * (1-actVal));
    }

    while(layerIdx > 0)
    {
        const numNeurons = LAYERS[layerIdx];

        for (let weightIdx=0; weightIdx < weights[layerIdx-1].length; weightIdx++)
        {
            let sum=0;
            for (let neuronIdx=0; neuronIdx < numNeurons; neuronIdx++)
            {
                const d = dEdZ[layerIdx][neuronIdx];
                const w = weights[layerIdx-1][weightIdx][neuronIdx];
                weightVector[layerIdx-1][weightIdx][neuronIdx] = d * activations[layerIdx-1][weightIdx];
                sum += d*w;
            }
            dEdZ[layerIdx-1][weightIdx] = sum  * sigmoidDerivatives[layerIdx-1][weightIdx];
        }

        layerIdx--;
    }

    debugger;
}


function test()
{
    const deltaWeight = .00000000001;

    for (let layerIdx=0; layerIdx < weights.length; layerIdx++)
    {
        document.write("<br/>");
        for (let inIdx=0; inIdx < weights[layerIdx].length; inIdx++)
        {
            for (let weightIdx=0; weightIdx < weights[layerIdx][inIdx].length; weightIdx++)
            {
                document.write("W(" + layerIdx + ")" + inIdx + "," + weightIdx + ".........");
                const weightsCopy = JSON.parse(JSON.stringify(weights));
                forwardPass([1,0]);
                const error = calcError([0,0,0]);
                weights[layerIdx][inIdx][weightIdx] += deltaWeight;
                forwardPass([1,0]);
                const deltaError = calcError([0,0,0]) - error;
                const derivative = deltaError/deltaWeight;
                document.write(derivative.toFixed(4) + " : " + weightVector[layerIdx][inIdx][weightIdx].toFixed(4) + "<br/>");
                weights = weightsCopy;
            }
        }
    }
}


function calcError(targetOutputs)
{
    let totalError = 0;
    for (let outIdx=0; outIdx < targetOutputs.length; outIdx++)
    {
        const diff = targetOutputs[outIdx] - activations[activations.length-1][outIdx];
        totalError +=  .5 * diff * diff;
    }
    return totalError;
}

init();

forwardPass([1,0]);
backPass([0,0,0]);

test();










