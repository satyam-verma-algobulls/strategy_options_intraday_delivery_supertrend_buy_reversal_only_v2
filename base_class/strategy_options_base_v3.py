from abc import abstractmethod
from datetime import time

import pandas
import tls

import clock
from segment import *
from strategy.core.strategy_options_base_v2 import StrategyOptionsBaseV2, OptionsStrikeDirection, OptionsInstrumentDirection, OptionsTradingsymbolSuffix, OrderTagManager
from strategy.utils import check_order_placed_successfully, HistoricalDataCacher, data_sanity_check, check_order_complete_status

LOCAL_CONSTANT_DAY_CANDLE = 1440


class StrategyOptionsBaseV3(StrategyOptionsBaseV2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.strategy_mode is StrategyMode.DELIVERY:
            self.strategy_dictionary = None
            self.memory_dictionary = None
            self.message_id = None
            self.STRATEGY_KEY = 'STRATEGY_KEY'

        self.candle_start_time = None
        self.candle_end_time = None
        self.order_tag_manager = None
        self.lot_size_map = self.get_lot_size_map()
        self.hc = HistoricalDataCacher()
        self._display_strategy_specific_details()
        self._candle_start_end_time()

        tls.TLS.subscription_manager_exit_all_positions_updates.subscribe(func=self.process_orders_on_exit_signal, threaded=False, insert_position_index=0)

    def initialize(self):
        super().initialize()
        self.order_tag_manager = OrderTagManager()

    def initialize_instrument(self, instrument):
        """
        Initializes the child instruments and expiry dates of the instruments that is passed
        For IDX instrument in yaml, call the method directly.
        For IDX instrument given as underlying_instrument (yaml has FUT or some other instrument through which child instruments cannot be extracted), pass self.underlying_instrument from the child class.
        """

        super().initialize_instrument(instrument)

    @abstractmethod
    def strategy_version(self):
        """
        Example:
            3.3.1 when a strategy is made for the first time
            3.3.2, 3.3.3 as changes, revisions, fixes, etc. are introduced in the strategy
        """

        raise NotImplementedError

    @abstractmethod
    def strategy_client_name(self):
        """
        Name of the Client as specified in the strategy document
        """

        raise NotImplementedError

    @abstractmethod
    def strategy_type(self):
        """
        Example:
            Options / Regular
        """

        raise NotImplementedError

    @staticmethod
    def versions_supported():
        return AlgoBullsEngineVersion.VERSION_3_3_0

    def _display_strategy_specific_details(self):
        """
        Displays Version, Type and Client Name at the start of the strategy
        """

        marker = "#"
        self.logger.info(f'\n{marker * 40}\nSTRATEGY VERSION: {self.strategy_version()}\n{marker * 40}')
        self.logger.debug(f'\n{marker * 60}\nSTRATEGY TYPE: {self.strategy_type()} | CLIENT: {self.strategy_client_name()}\n{marker * 60}')

    def base_candle_is_saved(self, base_instrument, base_candle_duration, base_candle_start_time_hours, base_candle_start_time_minutes, candle):
        """
        Fetches and saves the base instrument candle for the given hours, minutes and time period (duration)
        """
        if hasattr(self, 'base_candle'):
            base_inst_str = base_instrument.tradingsymbol
            if base_inst_str in self.base_candle:
                return True
            else:
                hist_data = self.hc.fetch_hist_data(base_instrument, candle, base_candle_duration)

                # Sanity check to see if data exist or not
                if not data_sanity_check(hist_data):
                    self.logger.debug(f'Hist data not found for {base_instrument} instrument - \n{hist_data}')
                    return False

                # Converting timestamp column to datetime type
                hist_data['timestamp'] = pandas.to_datetime(hist_data['timestamp'])

                # Checking hour, minute and date condition and storing it in val
                base_candle = hist_data.loc[
                    (hist_data['timestamp'].dt.hour == base_candle_start_time_hours) & (hist_data['timestamp'].dt.minute == base_candle_start_time_minutes) & (hist_data['timestamp'].dt.date == clock.CLOCK.now().date())]
                if base_candle.empty:
                    self.logger.info(f"No such candle exist")
                    return False
                else:
                    self.logger.info(f"Base candle is saved")
                    self.base_candle[base_inst_str] = base_candle.iloc[0]
                    return True
        else:
            self.logger.error(f'The following attribute not found: base_candle')
            raise AttributeError

    def _candle_start_end_time(self):
        """
        Creates candles for start and end time if the appropriate strategy parameters are passed through yaml
        """
        self.logger.debug(f"Delete this base class v3 reacher here 1")
        if hasattr(self, 'start_time_hours') and hasattr(self, 'start_time_minutes'):
            self.logger.debug(f"Delete this base class v3 reacher here ")
            try:
                self.candle_start_time = time(hour=self.start_time_hours, minute=self.start_time_minutes)
            except ValueError:
                self.logger.fatal('Error converting candle start time hours and minutes... EXITING')
                raise SystemExit

        if hasattr(self, 'end_time_hours') and hasattr(self, 'end_time_minutes'):
            try:
                self.candle_end_time = time(hour=self.end_time_hours, minute=self.end_time_minutes)
            except ValueError:
                self.logger.fatal('Error converting candle end time hours and minutes... EXITING')
                raise SystemExit

    def initialize_memory_dictionary(self):
        """
        Creates the memory dictionary
        Subclass has to call this method explicitly to use this functionality
        Note: Only used in memory management strategies
        """

        # Create memory dictionary
        if self.strategy_mode is StrategyMode.DELIVERY:
            _messages, success = self.get_messages(key=self.KEY)
            if not success:
                self.logger.error('Could not fetch messages!!!')
            print(_messages)
            if _messages:
                _message = _messages[0]
                self.memory_dictionary = _message['value']
                self.message_id = _message['id']
                self.logger.info(f'Fetched message: {_message} | {self.memory_dictionary} | {self.message_id}')
            else:
                self.logger.error(f'Message not found. Resetting! {_messages}')
                self.memory_dictionary = {}
                self.message_id = None

            # Sanity check
            if not isinstance(self.memory_dictionary, dict):
                self.logger.error('Fetched message is corrupt. Resetting!!!')
                self.memory_dictionary = {}
        else:
            self.memory_dictionary = {}
            self.message_id = None
        if self.message_id is None:
            self.logger.debug('Creating blank/dumb message dictionary for the first time')
            self.message_id, success = self.create_message(key=self.KEY, value=self.memory_dictionary)

    @staticmethod
    def get_lot_size_map():
        """
        Sets the default lot size for instruments
        Child class has to override this method in case the lot size is to be set as 1, 1 (example)
        """
        return {'NIFTY BANK': 36, 'NIFTY 50': 36}

    def get_underlying_instrument(self, instrument_segment, instrument_exchange_token):
        """
        Returns the instrument for the given segment and exchange token
        """

        segment_map = {1: SegmentNseEQ, 2: SegmentNseFO, 3: SegmentNseIDX, 4: SegmentMcxFO, 5: SegmentBseEQ, 6: SegmentBseFO, 7: SegmentNcdFO, 8: SegmentNyseEQ, 9: SegmentNasdaqEQ}

        segment, underlying_instrument = None, None

        exchange_token_to_tradingsymbol_map = {1035: 'INDIA VIX', 1001: 'NIFTY 50', 1016: 'NIFTY BANK', 1007: 'NIFTY FIN SERVICE'}

        try:
            segment = segment_map[instrument_segment]
            underlying_instrument = self.broker.get_instrument(segment=segment, tradingsymbol=exchange_token_to_tradingsymbol_map[instrument_exchange_token], master_data=True)

        except ValueError:
            self.logger.fatal(f'Could not create underlying instrument with segment {segment} and exchange token {instrument_exchange_token}, Kindly verify')
            raise SystemExit

        self.logger.debug(f"Underlying instrument {underlying_instrument}")
        return underlying_instrument

    def check_start_end_time(self):
        """
        This function checks if start and end time has hit or not
        """

        if clock.CLOCK.now().time() < self.candle_start_time:
            self.logger.debug(f'Start time not hit yet')
            return False

        if clock.CLOCK.now().time() >= self.candle_end_time:
            self.logger.debug(f'End time hit')
            return False

        return True

    def _check_price(self, instrument, price_value, price_string):
        """
        This function is used to check if trigger price or price values goes below zero and returns a numerical value

        instrument: obj of type Instrument
        _price_value: price of the order
        _price_string: string of the parameter
        """

        return_value = price_value
        if price_value is not None and price_value <= 0:
            self.logger.info(f'Price calculation has gone below zero, setting {price_string} to minimum tick size')
            return_value = instrument.tick_size
        return return_value

    def place_order_regular(self, instrument, trigger_price, price, variety, quantity, main_order=None, order_transaction_type=None, try_again=False):
        """ 
        This method is used to place both kinds of order, entry order & exit order (like stoploss order, target order, profit order, etc.)
        Some implementations require that the order placement be retried in case of any failure, this method handles that with the 'try_again' param.

        instrument: obj of type Instrument
        trigger_price: Trigger price of the order to be placed
        price: Limit price of the order to be placed
        variety: Variety of the order to be placed
        quantity: Quantity of the order to be placed
        main order: for entry orders main order is None, for exit orders main order is present on which exit order is placed
        order transaction type: for entry it is not None, for exit it gets the value from main order
        """

        trigger_price = self._check_price(instrument, trigger_price, 'trigger price')
        price = self._check_price(instrument, price, 'price')

        self.logger.info(f'Order | Instrument: {instrument} | Trigger price: {trigger_price} | Price: {price} | Qty: {quantity} | Variety: {variety}')

        # Here we set position , action of the incoming order
        position = BrokerExistingOrderPositionConstants.ENTER if main_order is None else BrokerExistingOrderPositionConstants.EXIT
        self.logger.debug(f"Delete this order_transaction_type {main_order.order_transaction_type if main_order is not None else None}")
        order_transaction_type = order_transaction_type if main_order is None else main_order.order_transaction_type
        place_order = self.broker.BuyOrderRegular if (order_transaction_type is BrokerOrderTransactionTypeConstants.BUY and position is BrokerExistingOrderPositionConstants.ENTER) or \
                                                     (order_transaction_type is BrokerOrderTransactionTypeConstants.SELL and position is BrokerExistingOrderPositionConstants.EXIT) else self.broker.SellOrderRegular
        _order = None
        try:
            _order = place_order(instrument=instrument, order_code=self.order_code, order_variety=variety, price=price, trigger_price=trigger_price, quantity=quantity, position=position, related_order=main_order)
            if _order.broker_order_id is None or _order.get_order_status() is self.broker.constants.BROKER_ORDER_STATUS_CONSTANTS.value.REJECTED:
                raise Exception
        except Exception as ex:
            if variety is not BrokerOrderVarietyConstants.MARKET and try_again:
                self.logger.info(f'Error while placing order, trying again with variety as MARKET order')
                self.logger.debug(f'Error while placing order, details:\n{ex}')
                variety = BrokerOrderVarietyConstants.MARKET
                _order = self.place_order_regular(instrument, trigger_price, price, variety, quantity, main_order, order_transaction_type)
            raise NotImplementedError
        return _order

    @abstractmethod
    def get_main_orders_particulars(self, instrument, sideband_info):
        """
        This functions returns trigger_price, price, variety and quantity to be used for placing the order

        instrument: can be used for getting instrument related details, like hist data
        sideband_info: will have details passed from strategy_select_instruments_for_entry method
        candle: can be used for hist data, which is further used for price, trigger_price, etc.
        """

        trigger_price = sideband_info['trigger_price'] if sideband_info['trigger_price'] else None
        price = sideband_info['price'] if sideband_info['price'] else None
        variety = sideband_info['variety'] if sideband_info['variety'] is not BrokerOrderVarietyConstants.MARKET else BrokerOrderVarietyConstants.MARKET
        qty = sideband_info['no_of_lots'] * instrument.lot_size

        return trigger_price, price, variety, qty

    @abstractmethod
    def save_entry_order_details(self, base_instrument, order, sideband_info, trigger_price, price, variety, qty, candle, entry_key):
        """
        This function is used to save the order and its details
        The details come from sideband_info as well as trigger_price, price, variety, qty

        order: the order to be stored
        sideband_info: will have details passed from strategy_select_instruments_for_entry method
        trigger_price: trigger price received from get_main_orders_particulars
        price: price received from get_main_orders_particulars
        variety: variety received from get_main_orders_particulars
        qty: quantity received from get_main_orders_particulars

        Candle is also passed, which is sometimes used to get hist_data and store some order related detail
        """

        base_inst_str = base_instrument.tradingsymbol
        self.order_tag_manager.add_order(order, tags=[base_inst_str, entry_key])

    def write_to_memory(self, message, _key, _data):
        """
        Save order related data to the memory so that it can be fetched the next day
        """

        base_inst_str = self.underlying_instrument.tradingsymbol
        self.strategy_dictionary[base_inst_str] = _data
        self.__revise_memory_management(message)

    def read_from_memory(self):
        """
        Retrieve data from the memory
        """

        if self.memory_dictionary is not None:
            self.strategy_dictionary = self.memory_dictionary[self.STRATEGY_KEY]
        else:
            self.logger.debug(f'Failed to get info from memory, exiting')
            raise SystemExit

    def __revise_memory_management(self, message=''):
        """
        Update the data in the memory
        """

        self.memory_dictionary[self.STRATEGY_KEY] = self.strategy_dictionary
        if self.strategy_mode is StrategyMode.DELIVERY:
            self.logger.info(message)
            _, success = self.update_message(self.message_id, self.KEY, self.memory_dictionary)
            if not success:
                self.logger.error('Could not update message')

    def options_instruments_set_up_local(self, base_instrument, tradingsymbol_suffix, current_close, direction=OptionsInstrumentDirection.EXACT):
        """
        Retrieve all the child instruments for the given base instrument, tradingsymbol suffix and current close price
        """

        expiry_dates = self.get_allowed_expiry_dates()
        for expiry_date in expiry_dates:
            self.logger.debug(f"Delete this options_instruments_set_up_local v3 {base_instrument}, {tradingsymbol_suffix}, {current_close}, {direction}")
            self.options_instruments_set_up(base_instrument, direction, expiry_date, tradingsymbol_suffix.value, current_close)

    def get_child_instrument(self, base_instrument, tradingsymbol_suffix, strike_direction, no_of_strikes_start, no_of_strikes_end, entry_threshold, index):
        """
        Returns the child instrument
        """

        if entry_threshold is None:

            # Get child instrument by number of strikes and strike direction
            expiry_date = self.get_allowed_expiry_dates()[0]
            child_instrument = self.get_options_instrument_with_strike_direction(base_instrument, expiry_date, tradingsymbol_suffix.value, strike_direction, no_of_strikes_start)
        else:

            # Get child instrument by threshold condition
            child_instrument = self.__get_child_instrument_by_threshold(base_instrument, tradingsymbol_suffix, strike_direction, no_of_strikes_start, no_of_strikes_end, entry_threshold, index)

        return child_instrument

    def __get_child_instrument_by_threshold(self, base_instrument, tradingsymbol_suffix, strike_direction, no_of_strikes_start, no_of_strikes_end, entry_threshold, index):
        """
        Fetch the CE/PE list based on the tradingsymbol_suffix, strike direction, no of strikes start and no of strikes end values.
        Get the latest close of each child instrument list.
        Return the child instrument that is closest or farthest from the threshold based on the index value.
        """

        instrument_dict = {}
        expiry_date = self.get_allowed_expiry_dates()[0]
        _key = self.get_options_ref_key(base_instrument, expiry_date)

        if tradingsymbol_suffix is OptionsTradingsymbolSuffix.CE:

            # Child instruments list map based on the strike direction value
            child_instrument_strike_map = {OptionsStrikeDirection.ITM.value: self.instruments_ce_itm[_key][-no_of_strikes_start:-no_of_strikes_end if no_of_strikes_end is not None else no_of_strikes_start],
                                           OptionsStrikeDirection.OTM.value: self.instruments_ce_otm[_key][no_of_strikes_start:no_of_strikes_end if no_of_strikes_end is not None else no_of_strikes_start],
                                           OptionsStrikeDirection.ATM.value: self.instruments_ce_itm[_key][-no_of_strikes_start:-no_of_strikes_end if no_of_strikes_end is not None else no_of_strikes_start] +
                                                                             [self.instruments_ce_atm[_key]] + self.instruments_ce_otm[_key][no_of_strikes_start:no_of_strikes_end if no_of_strikes_end is not None else no_of_strikes_start]}
        else:
            child_instrument_strike_map = {OptionsStrikeDirection.ITM.value: self.instruments_pe_itm[_key][no_of_strikes_start:no_of_strikes_end if no_of_strikes_end is not None else no_of_strikes_start],
                                           OptionsStrikeDirection.OTM.value: self.instruments_pe_otm[_key][-no_of_strikes_start:-no_of_strikes_end if no_of_strikes_end is not None else no_of_strikes_start],
                                           OptionsStrikeDirection.ATM.value: self.instruments_pe_itm[_key][no_of_strikes_start:no_of_strikes_end if no_of_strikes_end is not None else no_of_strikes_start] +
                                                                             [self.instruments_pe_atm[_key]] + self.instruments_pe_otm[_key][-no_of_strikes_start:-no_of_strikes_end if no_of_strikes_end is not None else no_of_strikes_start]}

        base_latest_close = self.broker.historical_data_feed.get_latest_ohlc(base_instrument)['close']

        self.logger.debug(f"{tradingsymbol_suffix} Base latest close {base_latest_close}  Calculated entry threshold {entry_threshold}")

        child_instrument_list = child_instrument_strike_map[strike_direction]

        self.logger.debug(f"{tradingsymbol_suffix} Child instrument list {child_instrument_list}")

        for child_instrument in child_instrument_list:

            # If data for a child instrument continue with the next child instrument in the list
            try:
                child_latest_close = self.broker.historical_data_feed.get_latest_ohlc(child_instrument)['close']
                self.logger.debug(f"{child_instrument} latest close {child_latest_close} | threshold condition {entry_threshold}")
                instrument_dict[child_instrument] = abs(child_latest_close - entry_threshold)

            except IndexError:
                self.logger.debug(f"Data not found for the instrument {child_instrument}")
                continue

        child_instrument = list({k: v for k, v in sorted(instrument_dict.items(), key=lambda item: item[1]) if v > 0})[index]

        return child_instrument

    def strategy_enter_position(self, candle, instrument, sideband_info):
        """
        This function executes the entry orders and can be overridden as per requirement
        candle: used to place the order
        instrument: instrument for which order will be placed
        sideband_info: details passed from strategy_select_instruments_for_entry method
        """

        trigger_price, price, variety, qty = self.get_main_orders_particulars(instrument, sideband_info)
        order_transaction_type = BrokerOrderTransactionTypeConstants.BUY if sideband_info['action'] is ActionConstants.ENTRY_BUY else BrokerOrderTransactionTypeConstants.SELL
        _order = self.place_order_regular(instrument, trigger_price, price, variety, qty, None, order_transaction_type)

        if check_order_placed_successfully(_order):
            base_instrument = sideband_info['base_instrument']
            entry_key = sideband_info['entry_key']
            self.save_entry_order_details(base_instrument, _order, sideband_info, trigger_price, price, variety, qty, candle, entry_key)
        return _order

    def get_exit_action(self, main_order):
        """
        Returns exit action based on the main order's transaction type
        Subclass has to call this method explicitly to use this functionality
        """
        if check_order_complete_status(main_order):
            return ActionConstants.EXIT_BUY if main_order.order_transaction_type is BrokerOrderTransactionTypeConstants.BUY else ActionConstants.EXIT_SELL
        self.logger.error(f'Main order is not complete. Exit action can be determined only for a complete order. Returning None...')


    def process_exits(self, main_order, main_order_complete, stoploss_order, stoploss_order_complete):
        """
        Process the stoploss and entry orders for exit
        """

        main_order_position_closed = False
        if stoploss_order:

            # If complete, main order position is closed
            if stoploss_order_complete:
                main_order_position_closed = True

            # If not complete, main order position is not closed, and cancel target order
            else:
                stoploss_order.cancel_order()
            self.order_tag_manager.remove_order(stoploss_order)

        if main_order:

            # Main order is complete
            if main_order_complete:

                # If not squared off with target/stoploss, exit position now
                if not main_order_position_closed:
                    self.place_exit_order(main_order.instrument, 0, 0, main_order.quantity, main_order, BrokerOrderVarietyConstants.MARKET)

            # Main order is not complete, cancel it
            else:
                main_order.cancel_order()

            # All done, now reset everything
            self.order_tag_manager.remove_order(main_order)

    def process_orders_on_exit_signal(self, *args, **kwargs):
        """
        Exit all open orders if an exit signal occurs.
        Example:
             Manual exit event is triggered
        Note:
            Method to be implemented only in order slicing strategies.
        """

        pass
