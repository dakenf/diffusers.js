import 'module-alias/register'

import { getModelFile } from '@/hub/node'
import { setCacheImpl } from '@/hub'

export { setModelCacheDir } from '@/hub/browser'

setCacheImpl({ getModelFile })
