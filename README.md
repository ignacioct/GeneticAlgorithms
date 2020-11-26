<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the GeneticAlgorithms and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** ignacioct, GeneticAlgorithms_name, twitter_handle, email, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/ignacioct/GeneticAlgorithms">
  </a>

  <h3 align="center">Genetic Algorithms </h3>

  <p align="center">
    Collection of some techniques of Evolutionary Computation implemented in Python
    <br />
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#algorithms">Algorithms</a></li>
      <ul>
        <li><a href="#genetic-algorithms">Genetic Algorithm</a></li>
      </ul>
      <ul>
        <li><a href="#evolutive-strategies">Evolutive Strategy</a></li>
      </ul>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is a collection of the techniques of Evolutionary Computation that I've been developing, for multiple purposes. For that reasons, the core functionalities of the algorithms are kept, but the fitness functions and its equivalents are kept *return 1*. This is for two reasons:
* Each application of a generic algorithm must have a purpose and a domain, hence each one will have its own fitness function
* The original function is not relevant to the documentation of the algorithm itself.

Other aspects of the algorithm that are relative to its original problem, like the codification and the population size, are being kept, in order to ilustrate how to implement your own. However, this aspect must also be personalized. 

A big part of my documentation in this subject consist on University lessons. However, I will try to keep the documentation section up to date with all the resources I find useful and relevant.


### Built With

* Python 3 (Compatible with all 3 subversions)

<!-- GETTING STARTED -->
## Documentation

* [An introduction to Genetic Algorithms](https://mitpress.mit.edu/books/introduction-genetic-algorithms), by Melanie Michelle
* [Python ThreadPoolExecutor Tutorial](https://tutorialedge.net/python/concurrency/python-threadpoolexecutor-tutorial/)
* [Elitism in GA: should I let the elites be selected as parents?](https://stackoverflow.com/questions/14622342/elitism-in-ga-should-i-let-the-elites-be-selected-
as-parents​ [Accessed 18 Oct. 2020].)
* [Genetic Algorithms via Random Offspring Generation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.106.8662&rep=rep1&type=pdf) by Miguel Rocha and José P. Neves


## Algorithms
### Genetic Algorithm

This implementation of an standard genetic algorithm was originally created to solve an optimization problem of air sensor in a big city: where, which and how should we place different air quality sensor to minimice the price and maximize the quality of the data obtained. We used to versions of the problem: a reduced one of 64 bits (16 different sensors, 4 stations) and a complete one of 384 (16 sensors, 24 stations). 

It encoded in binary strings, each group of 16 bits indicating a different station, and each position inside that group one of the different sensors that can be placed (1) or nor (0).

The initialization is uniformly random. The evaluation of the candidate population of solutions can be done sequentially or concurrently (made with Python's thread pool executor). The concurrent approach speeds up the execution of the code exponentially.

The developed type of selection is tournament, in two variants: with and without elitism. In the elist version, the best indiviuals are directly passed without competing, but the can still be selected in that competition.

The crossover takes two parents in the population selected by the tournament and creates two child solution from theirs via a random selection of the bits of the parents. For each position, a bit in the same position of one of the parents is chosen. 

One variant of this crossover techinque called Random Offspring Generation is also implemented. It is proposed and explained in the wonderful paper [*Genetic Algorithms via Random Offspring Generation*](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.106.8662&rep=rep1&type=pdf). 

The mutation creates random changes in the bit of each solution if a random variable is lower than a mutation factor, chosen as a parammeter.

### Evolutive Strategies

This second implementation of a evolutionary computation technique relies heavily on Python's Object Oriented Programming capabilities. 

The original implementation was created to solve a problem of a robot with 10 motors, which could operate between an angle of -180 and 180. A combination of angle which minimices a precision-based fitness function should have been reached. We worked with two problems, a reduced one with only 4 motors, and its complete version, with 10. 

The evolutive strategy relies on individuals with two vectors:
* A vector with the functional parts, the proposed values for the solution.
* A vector with variances, each one corresponding to each of the values of the functional part. A bigger variance value indicates that bigger changes must be made to reach the fitness function minima. 

The selection, crossover and mutation is made accordingly to the codification of real values, the value of the angles or the values of any other candidate problem can have decimals. 

Two evolutive strategies are implemented:
* One with only one individual in the population, which creates a child solution, and they compete to stay in the population for the next generation
* One with variable populations of individuals, closer to the ones developed in the genetic algorithms. 


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Ignacio Talavera Cepeda - [LinkedIn Profile](https://www.linkedin.com/in/ignacio-talavera-cepeda/) - ignaciotalaveracepeda@gmail.com

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/ignacioct/GeneticAlgorithms.svg?style=for-the-badge
[contributors-url]: https://github.com/ignacioct/GeneticAlgorithms/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ignacioct/GeneticAlgorithms.svg?style=for-the-badge
[forks-url]: https://github.com/ignacioct/GeneticAlgorithms/network/members
[stars-shield]: https://img.shields.io/github/stars/ignacioct/GeneticAlgorithms.svg?style=for-the-badge
[stars-url]: https://github.com/ignacioct/GeneticAlgorithms/stargazers
[issues-shield]: https://img.shields.io/github/issues/ignacioct/GeneticAlgorithms.svg?style=for-the-badge
[issues-url]: https://github.com/ignacioct/GeneticAlgorithms/issues
[license-shield]: https://img.shields.io/github/license/ignacioct/GeneticAlgorithms.svg?style=for-the-badge
[license-url]: https://github.com/ignacioct/GeneticAlgorithms/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/ignacioct
