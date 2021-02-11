const LAYERS = [2,3,3,1];

const activations = [];
const sigmoidDerivatives = [];
let weights = [];
const weightVector = [];
const biasVector = [];
let biases = [];
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
        biasVector[layerIdx] = [];
        dEdZ[layerIdx] = [];

        for (let neuronIdx=0; neuronIdx < numNeurons; neuronIdx++)
        {
            biases[layerIdx][neuronIdx] = -.5 + Math.random();
            biasVector[layerIdx][neuronIdx] = 0;
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
                    w.push(-.5 + Math.random());
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
        for (let inIdx = 0; inIdx < activations[layerIdx+1].length; inIdx++)
        {
            let z = 0;
            for (let outIdx = 0; outIdx < activations[layerIdx].length; outIdx++)
            {
                z += activations[layerIdx][outIdx] * weights[layerIdx][outIdx][inIdx];
            }
            z += biases[layerIdx+1][inIdx];
            const sigmoidVal = sigmoid(z);
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
                biasVector[layerIdx][neuronIdx] = d;
                sum += d*w;
            }
            dEdZ[layerIdx-1][weightIdx] = sum  * sigmoidDerivatives[layerIdx-1][weightIdx];
        }

        layerIdx--;
    }
}


function test(inputs, outputs)
{
    const delta = .0001;

    let s = "";
    for (let layerIdx=0; layerIdx < weights.length; layerIdx++)
    {
        s += "<br/>";
        for (let inIdx=0; inIdx < weights[layerIdx].length; inIdx++)
        {
            for (let weightIdx=0; weightIdx < weights[layerIdx][inIdx].length; weightIdx++)
            {
                s += ("W(" + layerIdx + ")" + inIdx + "," + weightIdx + ". . . . .");
                const weightsCopy = JSON.parse(JSON.stringify(weights));
                forwardPass(inputs);
                const error = calcError(outputs);
                weights[layerIdx][inIdx][weightIdx] += delta;
                forwardPass(inputs);
                const deltaError = calcError(outputs) - error;
                const derivative = deltaError/delta;
                s += (derivative.toFixed(8) + " : " + weightVector[layerIdx][inIdx][weightIdx].toFixed(8) + "<br/>");
                weights = weightsCopy;
            }

            s += ("B(" + layerIdx + ")" + inIdx + ". . . . .");
            const biasesCopy = JSON.parse(JSON.stringify(biases));
            forwardPass(inputs);
            const error = calcError(outputs);
            biases[layerIdx][inIdx] += delta;
            forwardPass(inputs);
            const deltaError = calcError(outputs) - error;
            const derivative = deltaError/delta;
            s += (derivative.toFixed(8) + " : " + biasVector[layerIdx][inIdx].toFixed(8) + "<br/>");
            biases = biasesCopy;
        }

    }

    return s;
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


function train(rate)
{
    for (let i=0; i < 1000; i++)
    {
        const batchSize = 1;
        const batchSizeReciprocal = 1 / batchSize;

        for (let batchIndex = 0; batchIndex < batchSize; batchIndex++)
        {
            const ins = generateIns();
            const outs = generateOuts(ins);
            forwardPass(ins);
            backPass(outs);
        }

        for (let layerIdx=0; layerIdx < weights.length; layerIdx++)
        {
            for (let inIdx = 0; inIdx < weights[layerIdx].length; inIdx++)
            {
                for (let weightIdx = 0; weightIdx < weights[layerIdx][inIdx].length; weightIdx++)
                {
                    weights[layerIdx][inIdx][weightIdx] += weightVector[layerIdx][inIdx][weightIdx] * batchSizeReciprocal * rate;
                    weightVector[layerIdx][inIdx][weightIdx] = 0;
                }
                biases[layerIdx][inIdx] += biasVector[layerIdx][inIdx] * batchSizeReciprocal * rate ;
                biasVector[layerIdx][inIdx] = 0;
            }
        }
    }
}


function generateIns()
{
    return [Number(Math.random() > .5), Number(Math.random() > .5)];
}


function generateOuts(ins)
{
    return [Number(ins[0] !== ins[1])];
}


init();




//train(.1);
//forwardPass([0,0]);


let cnt2 = 0;

document.getElementById('submitBut').onclick = () =>
{
    const inputs = document.getElementsByClassName('input');
    const inArr = [];
    for (let i = 0; i < inputs.length; i++)
    {
        const input = inputs[i];
        inArr.push(input.value);
    }

    forwardPass(inArr);
    backPass(generateOuts(inArr));
    let s = test(inArr, generateOuts(inArr));

    activations[activations.length-1].forEach( (v,i) => s += ("OUT " + i + " > " + v.toFixed(4) + "\n"));
    s += "<br/>Error : " + calcError(generateOuts(inArr));
    document.getElementById("result").innerHTML = s;


};



//forwardPass([1,0]);
//backPass([0]);
//test();

train(.1);

//debugger;

