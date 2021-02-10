

const activations =
[
    [],
    [0,0],
    [0,0]
];

const sigmoidDerivatives =
[
    [0,0],
    [0,0]
];

let weights =
[
    [
        [-.5,0],
        [.3,.7]
    ],
    [
        [.2,-.2],
        [.1,.8]
    ]
];

const weightVector =
[
    [
        [0,0],
        [0,0]
    ],
    [
        [0,0],
        [0,0]
    ]
];




const biases =
[
    [0, 0],
    [0, 0]
];


function initWeights()
{

}


function forwardPass(inputs)
{
    activations[0] = inputs;
    for (let layerIdx = 0; layerIdx < activations.length-1; layerIdx++)
    {
        for (let outIdx = 0; outIdx < activations[layerIdx+1].length; outIdx++)
        {
            let z = 0;
            for (let inIdx = 0; inIdx < inputs.length; inIdx++)
            {
                z += inputs[inIdx] * weights[layerIdx][outIdx][inIdx];
            }
            z += biases[layerIdx][outIdx];
            const sigmoidVal = sigmoid(z);
            activations[layerIdx+1][outIdx] = sigmoidVal;
            sigmoidDerivatives[layerIdx][outIdx] = sigmoidVal * (1 - sigmoidVal);
        }
        inputs = activations[layerIdx+1];
    }

}


function sigmoid(x)
{
    return 1 / (1 + Math.exp(-x));
}


function backPass(targetOutputs)
{
    const tmpVals = [];
    let layerIdx = activations.length - 2;

    let totalError = 0;

    for (let outIdx=0; outIdx < targetOutputs.length; outIdx++)
    {
        const diff = targetOutputs[outIdx] - activations[layerIdx+1][outIdx];
        totalError +=  .5 * diff * diff;
        tmpVals[outIdx] = diff;
    }

    for (let outIdx=0; outIdx < weightVector[layerIdx].length; outIdx++)
    {
        tmpVals[outIdx] *= sigmoidDerivatives[layerIdx][outIdx];

        for (let weightIdx = 0; weightIdx < weightVector[layerIdx][outIdx].length; weightIdx++)
        {
            weightVector[layerIdx][outIdx][weightIdx] += tmpVals[outIdx] * activations[layerIdx][weightIdx];
        }
    }

    layerIdx--;

    weightVector[0][0][0] +=
        tmpVals[0] * weights[1][0][0] * sigmoidDerivatives[0][0] * activations[0][0] +
        tmpVals[1] * weights[1][1][0] * sigmoidDerivatives[0][0] * activations[0][0];

    weightVector[0][0][1] +=
        tmpVals[0] * weights[1][0][0] * sigmoidDerivatives[0][0] * activations[0][1] +
        tmpVals[1] * weights[1][1][0] * sigmoidDerivatives[0][0] * activations[0][1];

    weightVector[0][1][0] +=
        tmpVals[0] * weights[1][0][1] * sigmoidDerivatives[0][1] * activations[0][0] +
        tmpVals[1] * weights[1][1][1] * sigmoidDerivatives[0][1] * activations[0][0];

    weightVector[0][1][1] +=
        tmpVals[0] * weights[1][0][1] * sigmoidDerivatives[0][1] * activations[0][1] +
        tmpVals[1] * weights[1][1][1] * sigmoidDerivatives[0][1] * activations[0][1];

    // debugger;

    // document.write("Error : " + totalError + "<br/><br/>");
}


// forwardPass([1,1]);
// backPass([0,1]);



// test();
// document.write("Error : " + calcError([0,1]));
train();
forwardPass([0,1]);
debugger;

function test()
{
    const deltaWeight = .00000001;
    for (let layerIdx=0; layerIdx < weights.length; layerIdx++)
    {
        for (let inIdx=0; inIdx < weights[layerIdx].length; inIdx++)
        {
            for (let weightIdx=0; weightIdx < weights[layerIdx][inIdx].length; weightIdx++)
            {
                document.write("<br/>Derivative of " + weights[layerIdx][inIdx][weightIdx] + "<br/>");
                const weightsCopy = JSON.parse(JSON.stringify(weights));
                forwardPass([1,1]);
                const error = calcError([0,1]);
                weights[layerIdx][inIdx][weightIdx] += deltaWeight;
                forwardPass([1,1]);
                const deltaError = calcError([0,1]) - error;
                const derivative = deltaError/deltaWeight;
                document.write(derivative + " vs " + weightVector[layerIdx][inIdx][weightIdx] + "<br/>");
                weights = weightsCopy;
            }
        }


    }
}

function train()
{
    for (let i=0; i<10000; i++)
    {
        const batchSize = 5;

        for (let batchIndex = 0; batchIndex < batchSize; batchIndex++)
        {
            const ins = [];
            const outs = [];
            let inBool = Math.random() > .5;
            ins[0] = Number(inBool);
            inBool = Math.random() > .5;
            ins[1] = Number(inBool);

            if (ins[0] !== ins[1])
            {
                outs[0] = 1;
                outs[1] = 1;
            }
            else
            {
                outs[0] = 0;
                outs[1] = 0;
            }

            forwardPass(ins);
            backPass(outs);
        }

        for (let layerIdx=0; layerIdx < weights.length; layerIdx++)
        {
            for (let inIdx = 0; inIdx < weights[layerIdx].length; inIdx++)
            {
                for (let weightIdx = 0; weightIdx < weights[layerIdx][inIdx].length; weightIdx++)
                {
                    weights[layerIdx][inIdx][weightIdx] += ((weightVector[layerIdx][inIdx][weightIdx] / batchSize) * .3);
                    weightVector[layerIdx][inIdx][weightIdx] = 0;
                }
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





