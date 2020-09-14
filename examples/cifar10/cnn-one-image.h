#pragma once

#include <string>

#include "singa/model/feed_forward_net.h"
#include "singa/model/optimizer.h"
#include "singa/model/metric.h"
#include "singa/utils/channel.h"
#include "singa/utils/string.h"
#include "singa/io/snapshot.h"

namespace singa {

const string snapshot_name="mysnap-30epoch";

int Eval(string file);

}
