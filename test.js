const NeuronalNetwork = require('./neuronalNetwork.js');

let brain = new NeuronalNetwork(2, [5, 5], 1, 0.001);
//brain.loadWeigthsTrained();
console.log(Math.round(brain.feedForward([1, 1])));
console.log(Math.round(brain.feedForward([0, 0])));
console.log(Math.round(brain.feedForward([0, 1])));
console.log(Math.round(brain.feedForward([1, 0])));

console.log('----------------------');

/*
brain.loadWeigthsTrained();
for (let i = 0; i < 2000; i++) {
    brain.train([3, 3], [3]);
}
brain.saveWeigths();

console.log(Math.round(brain.feedForward([1, 1])));
console.log(Math.round(brain.feedForward([0, 0])));
console.log(Math.round(brain.feedForward([0, 1])));
console.log(Math.round(brain.feedForward([1, 0])));

console.log(Math.round(brain.feedForward([3, 3])));*/