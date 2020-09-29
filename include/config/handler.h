/*
	co-ecm
	Copyright (C) 2018  Jonas Wloka

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
			the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
			but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __CONFIG_HANDLER_H__
#define __CONFIG_HANDLER_H__

#include "config/ini.h"

/**
 * Configuration handler.
 *
 * Called by ini.h to set program values to configuration file values.
 *
 * @param user Configuration
 * @param section ini-file section
 * @param name ini-file value name
 * @param value ini-file value
 * @return
 */
int config_handler(void* user, const char* section, const char* name, const char* value);

#endif /* __CONFIG_HANDLER_H__ */
