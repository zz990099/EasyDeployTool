#pragma once

#include <iostream>

namespace easy_deploy {

inline void progress_bar(size_t cur, size_t total)
{
  int   bar_width = 50;
  float progress  = (float)cur / total;
  std::cout << "\r[";
  int pos = bar_width * progress;
  for (int i = 0; i < bar_width; ++i)
  {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << "% (" << cur << "/" << total << ")" << std::flush;
}

} // namespace easy_deploy
