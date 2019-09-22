const Matrix = require('./matrix.js');
const fs = require('fs');

class NeuronalNetwork {
    /**
     * Constructor. 
     * @param {int} inputs 
     * @param {int} hiddens       
     * @param {int} outputs 
     */
    constructor(inputs, hiddens, outputs, learningRate) {
        this.hiddensL = hiddens.length; //length
        this.wih = [];
        this.biasH = [];
        for (let i = 0; i < this.hiddensL; i++) {
            if (i == 0) {
                this.wih[i] = new Matrix(hiddens[i], inputs);
                this.biasH[i] = new Matrix(hiddens[i], 1);
            } else {
                this.wih[i] = new Matrix(hiddens[i], hiddens[i - 1]);
                this.biasH[i] = new Matrix(hiddens[i], 1);
            }
        }
        //this.saveWeigths(); // Save in the database
        //this.wih = new Matrix(hidden, inputs);
        this.who = new Matrix(outputs, hiddens[this.hiddensL - 1]);

        this.biasO = new Matrix(outputs, 1);
        this.learningRate = learningRate;
    }

    /**
     * feed forward with back propagation.
     * @param {Array} inp 
     */
    feedForward(inp) {
        console.log(inp)
            // convert matrix from array.
            // Load weigths
        this.loadWeigthsTrained();

        let input = this.wih[0].matrixFromArray(inp);
        let resultInputHidden = [];

        for (let i = 0; i < this.hiddensL; i++) {
            input = (i == 0) ? input : resultInputHidden[i - 1];
            // first step
            resultInputHidden[i] = this.wih[i].multiply(input);
            resultInputHidden[i].add(this.biasH[i]);
            resultInputHidden[i].addFunctionActivation();
        }
        // second step
        let resultHiddenOutputs = this.who.multiply(resultInputHidden[this.hiddensL - 1]);
        resultHiddenOutputs.add(this.biasO);
        resultHiddenOutputs.addFunctionActivation();
        let result = resultHiddenOutputs.toArray();
        return result;
    }

    /**
     * Train the neuronal network
     * @param {Array} inputs 
     * @param {Array} target 
     */
    train(inputs, target) {
        // Load weigths
        // this.loadWeigthsTrained();

        //convert array to matrix.
        let inputM = this.who.matrixFromArray(inputs);
        let targetM = this.who.matrixFromArray(target);
        let resultInputHidden = [];

        //Multiply weigths with the nodes.
        for (let i = 0; i < this.hiddensL; i++) {
            let input = (i == 0) ? inputM : resultInputHidden[i - 1];
            // first step
            resultInputHidden[i] = this.wih[i].multiply(input);
            resultInputHidden[i].add(this.biasH[i]);
            resultInputHidden[i].addFunctionActivation();
        }

        // Multiply the output with the last weigths and get the error
        let output = this.who.multiply(resultInputHidden[this.hiddensL - 1]);
        output.add(this.biasO);
        output.addFunctionActivation();
        //Target - Outputs
        let outputError = targetM.substract(output);

        // Back propagation algorithm.

        //Get gradient output to hidden
        let gradientOH = output.derivateFunction();
        gradientOH.getGradient(outputError, this.learningRate);

        //Calculate Deltas
        let hiddenTranspose = resultInputHidden[this.hiddensL - 1].transpose();
        let weightsHODeltas = gradientOH.multiply(hiddenTranspose);

        //adjust weights hidden - outputs and bias
        this.who.add(weightsHODeltas);
        this.biasO.add(gradientOH);

        //Variables.
        let whoTranspose;
        let error = [];
        let aux = this.hiddensL - 2;
        let inputTranspose = [];

        for (let i = this.hiddensL - 1; i >= 0; i--) {
            if (i == this.hiddensL - 1) {
                //console.log('Calcular error de salida ' + i)
                //Calculate error using the last weigths
                whoTranspose = this.who.transpose();
                error[i] = whoTranspose.multiply(outputError);
            } else {
                //console.log('Calcular error ' + i)
                //Calculate error using the last error
                whoTranspose = this.wih[i + 1].transpose();
                error[i] = whoTranspose.multiply(error[i + 1]);
            }
            //console.log('Calculamos el gradiente ' + i)
            //Get the gradient using the error
            let gradient = resultInputHidden[i].derivateFunction();
            gradient.getGradient(error[i], this.learningRate); // Gradient * outputError *learningRate = new value
            // In this part check if is the last part of the propagation if is the last part use the input for 
            // get the gradient.
            if (aux < 0) {
                inputTranspose[i] = inputM.transpose();
            } else {
                inputTranspose[i] = resultInputHidden[aux].transpose();
                aux--;
            }
            // Update the weigths.
            let weigthsDeltas = gradient.multiply(inputTranspose[i]);
            //console.log('Actualizamos pesos ' + i)
            // adjust the bias
            this.wih[i].add(weigthsDeltas);
            this.biasH[i].add(gradient);
        }
        //Save weigths
        //this.saveWeigths();
    }

    /**
     * Save in a json file.
     */
    saveWeigths() {
        //Prepare variable for json file.
        let obj = {
            weigths: [],
            bias: [],
            who: [], //weigths output
            biasO: []
        };
        for (let i = 0; i < this.hiddensL; i++) {
            obj.weigths.push(this.wih[i]);
            obj.bias.push(this.biasH[i]);
        }
        //Save weigths output.
        obj.who.push(this.who);
        obj.biasO.push(this.biasO);

        //Save in a json file
        let json = JSON.stringify(obj);
        try {
            fs.writeFileSync('weigths.json', json);
        } catch (error) {
            console.error('Error');
        }

    }

    /**
     * Load weights from json file.
     */
    loadWeigthsTrained() {
        try {
            let info = fs.readFileSync('weigths.json', 'utf8');
            let data = JSON.parse(info);
            for (let i = 0; i < data.weigths.length; i++) {
                let row = data.weigths[i].rows;
                let col = data.weigths[i].cols;
                this.wih[i] = new Matrix(row, col, data.weigths[i].matrix);
                this.biasH[i] = new Matrix(row, 1, data.bias[i].matrix);
            }
            let row = data.who[0].rows;
            let col = data.who[0].cols;
            this.who = new Matrix(row, col, data.who[0].matrix);
            this.biasO = new Matrix(row, 1, data.biasO[0].matrix)
        } catch (error) {
            console.error('Error file not found!');
        }
    }

    /**
     * Test percentage result from neuronal network.
     * @param {map} dataTest 
     */
    testResult(dataTest) {
        let result = 0;
        let aux = 0;
        let percentage;
        for (data of dataTest) {
            aux++;
            if (this.checkResult(data.targets, data.inputs)) {
                result++;
            } else {
                console.log('Fallo')
            }
        }

        percentage = (result / aux) * 100;
        console.log('Porcentaje de acierto')
        console.log(percentage + '%');
    }

    /**
     *  Check the result output
     * @param {Array} result 
     * @param {Array} input 
     */
    checkResult(result, input) {
        let aux = 0;
        let dataTrainined = this.feedForward(input);
        for (let i = 0; i < result.length; i++) {
            if (result[i] == Math.round(dataTrainined[i])) {
                aux++;
            }
        }
        if (result.length == aux) {
            return true;
        } else {
            return false;
        }
    }
}

module.exports = NeuronalNetwork;