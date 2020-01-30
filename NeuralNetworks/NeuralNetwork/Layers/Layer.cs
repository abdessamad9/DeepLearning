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
    public interface Layer : ILayer
    {

        LayerType Type { get; }
        /*int layerSize;
      

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

       



        public void Propagate(Matrix<double> input)
            {

            }


        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {

        }

  
        public void UpdateParameters()
        {

        }

        Matrix<double> activation;
        public  Matrix<double> Activation
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

        public bool Equals(ILayer other)
        {
            return LayerSize == other.LayerSize && InputSize == other.InputSize && BatchSize == other.BatchSize;
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
        }*/

    }
}
