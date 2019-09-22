const NeuronalNetwork = require('./neuronalNetwork.js');

//Layers and learning Rate
let brain = new NeuronalNetwork(2, [5, 5], 1, 0.001);
let trainingData = [{
        inputs: [1, 1],
        targets: [2]
    },
    {
        inputs: [0, 1],
        targets: [1]
    },
    {
        inputs: [0, 0],
        targets: [2]
    },
    {
        inputs: [1, 0],
        targets: [1]
    },
];


for (let i = 0; i < 10000; i++) {
    for (data of trainingData) {
        brain.train(data.inputs, data.targets);

    }
}
brain.saveWeigths();
brain.testResult(trainingData);