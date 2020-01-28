using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Layers;

using System;

namespace NeuralNetwork
{
    public sealed class Network : IEquatable<Network>, INetwork
    {


        public int batchSize;
        
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

        public Network (int batchsize, int numberLayer){
           BatchSize = batchsize;
           this.Layers = new ILayer[numberLayer];



        }


        Matrix<double> output;
        public Matrix<double> Output
        {
            get
            {
                return output;
            }
             set
            {
                this.output = value;
            }
        }

        ILayer[] layers;
        public ILayer[] Layers
        {
            get
            {
                return layers;
            }
            set
            {
                this.layers = value;
            }
        }

        Mode mode;
        public Mode Mode
        {
            get
            {
                return mode;
            }
            set
            {
                this.mode = value;
            }
        }



        public bool Equals(Network other)
        {

            /*  if (Layers.Length != other.Layers.Length)
              {
                  return false;
              }
              bool res;
              for (  int i =0; i< Layers.Length; i++)
              {
                  res = res && Layers[i].Equals(other.Layers[i]);
              }*/
            return BatchSize == other.BatchSize && Mode == other.Mode && Output == other.Output && Mode == other.Mode && Layers == other.Layers;

        }

        public void Learn(Matrix<double> outputLayerError)
        {
            var Input = outputLayerError;
            for (int i = Layers.Length - 1; i >= 0; i--)
            {
                Layers[i].BackPropagate(Input);
                Layers[i].UpdateParameters();
                Input = Layers[i].WeightedError; // W*B
            }
        }

        public void Propagate(Matrix<double> input)
        {
            Layers[0].Propagate(input);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].Propagate(Layers[i-1].Activation);
            }
            Output = Layers[Layers.Length - 1].Activation;
        }
    }
}