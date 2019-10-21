# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        
        listGhost = successorGameState.getGhostPositions()
        for ghost in listGhost:
          disFromGhost = manhattanDistance(newPos, ghost)
          if (disFromGhost <=1 ):
            return float("-inf")

        listFood = newFood.asList()
        listDisFood = []

        if(len(listFood) == 0):
          return float("inf")

        for food in listFood:
          disFood = manhattanDistance(newPos, food)
          listDisFood.append(disFood)

        minDisFood = min(listDisFood)

        if(len(listFood) < currentGameState.getNumFood()):
          return float("inf")

        score = (10*(1/minDisFood))
        return score



def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        numberOfGhosts = gameState.getNumAgents() - 1

        def maxLevel(gameState, depth):
          curDepth = depth + 1
          if gameState.isWin() or gameState.isLose() or curDepth == self.depth:
            return self.evaluationFunction(gameState)
          maxValue = float("-inf")
          actions = gameState.getLegalActions(0)
          for action in actions:
            succ = gameState.generateSuccessor(0,action)
            maxValue = max(maxValue,minLevel(succ,curDepth,1))
          return maxValue

        def minLevel(gameState, depth, agentIndex):
          minValue = float("inf")
          if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
          actions = gameState.getLegalActions(agentIndex)
          for action in actions:
            succ = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == (gameState.getNumAgents() - 1):
              minValue = min(minValue, maxLevel(succ,depth))
            else:
              minValue = min(minValue, minLevel(succ,depth,agentIndex+1))
          return minValue

        actions = gameState.getLegalActions(0)
        curScore = float("-inf")
        returnAction = ''
        for action in actions:
          nextState = gameState.generateSuccessor(0,action)
          score = minLevel(nextState,0,1)
          if score > curScore:
            returnAction = action
            curScore = score
        return returnAction
     

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        totalAgents = gameState.getNumAgents()

        def maxLevel(gameState, curDepth, alpha, beta):
          actions = gameState.getLegalActions(0)
          if curDepth > self.depth or gameState.isWin() or not actions:
            return self.evaluationFunction(gameState), Directions.STOP
          maxValue = float("-inf")
          bestAction = Directions.STOP
          for action in actions:
            succ = gameState.generateSuccessor(0, action)
            score = minLevel(succ, 1, curDepth, alpha, beta)[0]
            if score > maxValue:
              maxValue = score
              bestAction = action
            if maxValue > beta:
              return maxValue, bestAction
            alpha = max(alpha,maxValue)
          return maxValue, bestAction

        def minLevel(gameState, agentIndex, curDepth, alpha, beta):
          actions = gameState.getLegalActions(agentIndex)
          if not actions or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
          minValue = float("inf")
          bestAction = Directions.STOP
          flag = agentIndex == totalAgents - 1

          for action in actions:
            succ = gameState.generateSuccessor(agentIndex, action)
            if flag:
              score = maxLevel(succ, curDepth + 1, alpha, beta)[0]
            else:
              score = minLevel(succ, agentIndex + 1, curDepth, alpha, beta)[0]

            if score < minValue:
              minValue = score
              bestAction = action
            if minValue < alpha:
              return minValue, bestAction
            beta = min(beta, minValue)
          return minValue, bestAction
        
        defaultAlpha = float("-inf")
        defaultBeta = float("inf")

        return maxLevel(gameState, 1, defaultAlpha, defaultBeta)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        totalAgents = gameState.getNumAgents()

        def maxLevel(gameState, curDepth):
          actions = gameState.getLegalActions(0)
          if curDepth > self.depth or gameState.isWin() or not actions:
            return self.evaluationFunction(gameState), None

          succScore = []
          for action in actions:
            succ = gameState.generateSuccessor(0, action)
            succScore.append((minLevel(succ, 1, curDepth)[0], action))

          return max(succScore)

        def minLevel(gameState, agentIndex, curDepth):
          actions = gameState.getLegalActions(agentIndex)
          if not actions or gameState.isLose():
            return self.evaluationFunction(gameState), None
          
          successsors = [gameState.generateSuccessor(agentIndex, action) for action in actions]
          succScore = []
          flag = agentIndex == totalAgents - 1

          for succ in successsors:
            if flag:
              succScore.append(maxLevel(succ, curDepth + 1))
            else:
              succScore.append(minLevel(succ, agentIndex + 1, curDepth))
          
          averageScore = sum(map(lambda x: float(x[0]) / len(succScore), succScore))
          return averageScore, None
        return maxLevel(gameState, 1)[1]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

