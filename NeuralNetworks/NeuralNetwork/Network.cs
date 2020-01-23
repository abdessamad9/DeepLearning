using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork
{
    public sealed class Network : IEquatable<Network>, INetwork
    {

        public int BatchSize
        {
            get
            {
                return BatchSize;
            }
            set
            {
                this.BatchSize = value;
            }

        }

        public Matrix<double> Output
        {
            get
            {
                return Output;
            }
        }

        public ILayer[] Layers
        {
            get
            {
                return Layers;
            }
            set
            {
                this.Layers = value;
            }
        }

        public Mode Mode
        {
            get
            {
                return Mode;
            }
            set
            {
                this.Mode = value;
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
            for (int i = Layers.Length - 1; i >= 0; i--)
            {
                Layers[i].BackPropagate(outputLayerError);
            }
        }

        public void Propagate(Matrix<double> input)
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].Propagate(input);
            }
        }
    }
}