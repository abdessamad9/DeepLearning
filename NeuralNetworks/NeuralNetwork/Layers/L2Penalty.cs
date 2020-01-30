﻿using System;
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
    public class L2Penalty : ILayer
    {
        //
        public L2Penalty(ILayer layer, double indiceLayer, double penaltyCoefficient)
        {


            UnderlyingLayer = layer;
            IndiceLayer = indiceLayer;

            PenaltyCoefficient = penaltyCoefficient;
        }

        double penaltyCoefficient;
        public double PenaltyCoefficient
        {
            get
            {
                return penaltyCoefficient;
            }
            set
            {
                this.penaltyCoefficient = value;
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
        double indiceLayer;
        public double IndiceLayer
        {
            get
            {
                return indiceLayer;
            }
            set
            {
                this.indiceLayer = value;
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

        public bool Equals(ILayer other)
        {
            return UnderlyingLayer.Equals(other) ;
        }

        public void Propagate(Matrix<double> input)
        {
            UnderlyingLayer.Propagate(input);
        }

        public void UpdateParameters()
        {
            //Vector<double> res;
           double coefficient = ((Standard)UnderlyingLayer).Computation();
            ((Standard)UnderlyingLayer).UpdateBias(((Standard)UnderlyingLayer).VelocityBias, 1-coefficient*PenaltyCoefficient);
            ((Standard)UnderlyingLayer).UpdateWeights(((Standard)UnderlyingLayer).VelocityWeights, 1 - coefficient * PenaltyCoefficient);
        }
        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            UnderlyingLayer.BackPropagate(upstreamWeightedErrors);

        }


    }
}


