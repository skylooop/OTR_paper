import time
from typing import Optional

from .video import VideoRecorder
import acme
from acme import core
from acme.utils import counting
from acme.utils import loggers
import dm_env


class D4RLEvalLoop(core.Worker):

  def __init__(
      self,
      environment: dm_env.Environment,
      actor: acme.Actor,
      label: str = "evaluation",
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
  ):
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(label)
    #self._video = VideoRecorder("/home/m_bobrin/optimal_transport_reward/otr", fps=20)
  
  def run(self, num_episodes: int):  # pylint: disable=arguments-differ
    # Update actor once at the start
    self._actor.update(wait=True)
    total_episode_return = 0.0
    total_episode_steps = 0
    start_time = time.time()

    #self._video.init(enabled=True)
    
    for _ in range(num_episodes):
      timestep = self._environment.reset()
      #self._video.record(self._environment)
      
      self._actor.observe_first(timestep)
      while not timestep.last():
        action = self._actor.select_action(timestep.observation)
        timestep = self._environment.step(action)
        #if _ < 2:
          #self._video.record(self._environment)
        self._actor.observe(action, timestep)
        total_episode_steps += 1
        total_episode_return += timestep.reward
    
      #self._video.save(f"video/eval_test.mp4")
    steps_per_second = total_episode_steps / (time.time() - start_time)
    counts = self._counter.increment(
        steps=total_episode_steps, episodes=num_episodes)
    average_episode_return = total_episode_return / num_episodes
    average_episode_steps = total_episode_steps / num_episodes
    average_normalized_return = self._environment.get_normalized_score(
        average_episode_return)
    result = {
        "average_episode_return": average_episode_return,
        "average_normalized_return": average_normalized_return,
        "average_episode_length": average_episode_steps,
        "steps_per_second": steps_per_second,
    }
    result.update(counts)
    self._logger.write(result)
