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
    public class InputStadardizing : Layer
    {
        public InputStadardizing(ILayer layer, double[] mean, double[] stdDev)
        {
            UnderlyingLayer = layer;
            Mean = Matrix<double>.Build.Dense(this.underlyingLayer.InputSize, this.underlyingLayer.BatchSize,0.0 );
            StdDev = Matrix<double>.Build.Dense(this.underlyingLayer.InputSize, this.underlyingLayer.BatchSize, 0.0);
            for (int i=0; i < this.underlyingLayer.BatchSize; i++)
            {
                Mean.SetColumn(i, mean);
                StdDev.SetColumn(i, stdDev);
            }
        }

        Matrix<double> mean;
        public Matrix<double> Mean
        {
            get
            {
                return mean;
            }
            set
            {
                this.mean = value;
            }
        }
        Matrix<double> stdDev;
        public Matrix<double> StdDev
        {
            get
            {
                return stdDev;
            }
            set
            {
                this.stdDev = value;
            }
        }

        ILayer underlyingLayer;
        public ILayer UnderlyingLayer
        {
            get
            {
                return underlyingLayer;
            }
            set
            {
                this.underlyingLayer = value;
            }
        }

        public int LayerSize
        {
            get
            {
                return ((Standard)UnderlyingLayer).LayerSize;
            }
            set
            {
                ((Standard)UnderlyingLayer).LayerSize = value;
            }

        }

        public int InputSize
        {
            get
            {
                return ((Standard)UnderlyingLayer).InputSize;
            }
            set
            {
                ((Standard)UnderlyingLayer).InputSize = value;
            }

        }

        public int BatchSize
        {
            get
            {
                return ((Standard)UnderlyingLayer).BatchSize;
            }
            set
            {
                ((Standard)UnderlyingLayer).BatchSize = value;
            }

        }
        public Matrix<double> Activation
        {
            get
            {
                return ((Standard)UnderlyingLayer).Activation;
            }
            set
            {
                ((Standard)UnderlyingLayer).Activation = value;
            }

        }
        public Matrix<double> WeightedError
        {
            get
            {
                return ((Standard)UnderlyingLayer).WeightedError;
            }
            set
            {
                ((Standard)UnderlyingLayer).WeightedError = value;
            }

        }

        LayerType type;
        public LayerType Type
        {
            get
            {
                return LayerType.InputStandardizing;
            }
        }

        public bool Equals(ILayer other)
        {
            return UnderlyingLayer.Equals(other);
        }

        public void Propagate(Matrix<double> input)
        {
            /*Vector<double> m = input.RowSums();
            m.Divide(input.ColumnCount,m);
            Vector<double> std = input.RowSums();
            for (int i = 0; i < input.ColumnCount; i++)
            {
                std.Add(input.Column(i) * input.Column(i));
            }
            std.Add(-m*m);
            for (int i = 0; i < input.ColumnCount; i++)
            {
                input.SetColumn(i,input.Column(i) - m);
            }
            input.Subtract(Mean,input);
            input.PointwiseDivide(StdDev, input);
            UnderlyingLayer.Propagate(input);*/
        }

        public void UpdateParameters()
        {
            double coefficient = ((Standard)UnderlyingLayer).Computation();
            ((Standard)UnderlyingLayer).UpdateBias(((Standard)UnderlyingLayer).VelocityBias, 1);
            ((Standard)UnderlyingLayer).UpdateWeights(((Standard)UnderlyingLayer).VelocityWeights,1);
        }
        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            UnderlyingLayer.BackPropagate(upstreamWeightedErrors);

        }
    }
}
