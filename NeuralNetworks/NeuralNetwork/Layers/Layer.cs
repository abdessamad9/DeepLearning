using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.GradientAdjustmentsParameters;
using NeuralNetwork.Common.Layers;


namespace NeuralNetwork.Layers
{
    public class Layer : ILayer
    {
       
        int layerSize;
        public int LayerSize
        {
            get
            {
                return layerSize;
            }
            set
            {
                this.layerSize = value;
            }
        }
        int inputSize;
        public int InputSize
        {
            get
            {
                return inputSize;
            }
            set
            {
                this.inputSize = value;
            }
        }
     

        int batchSize;
        public int BatchSize
        {
            get
            {
                return batchSize;
            }
            set
            {
                this.batchSize = value;
            }
        }

        LayerType Type { get; set; }



        void Propagate(Matrix<double> input)
            {

            }


        void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {

        }

  
        void UpdateParameters()
        {

        }

        Matrix<double> activation;
        public Matrix<double> Activation
        {
            get
            {
                return activation;
            }
            set
            {
                this.activation = value;
            }
        }

        Matrix<double> weightedError;
        public Matrix<double> WeightedError
        {
            get
            {
                return weightedError;
            }
            set
            {
                this.weightedError = value;
            }
        }

    }
}
